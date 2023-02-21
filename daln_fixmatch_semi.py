import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from tllib.modules.classifier import Classifier
from tllib.self_training.pseudo_label import ConfidenceBasedSelfTrainingLoss
from tllib.vision.transforms import MultipleApply
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

from daln.nwd import NuclearWassersteinDiscrepancy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageClassifier(Classifier):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim=1024, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

    def forward(self, x: torch.Tensor):
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        predictions = self.head(f)
        if self.training:
            return predictions, f
        else:
            return predictions


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data Loading Codes
    train_source_transform = utils.get_train_transform(args.train_resizing, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                                                       random_horizontal_flip=not args.no_hflip,
                                                       random_color_jitter=False, resize_size=args.resize_size,
                                                       norm_mean=args.norm_mean, norm_std=args.norm_std)
    weak_augment = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                             random_horizontal_flip=not args.no_hflip,
                                             random_color_jitter=False, resize_size=args.resize_size,
                                             norm_mean=args.norm_mean, norm_std=args.norm_std)
    strong_augment = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                               random_horizontal_flip=not args.no_hflip,
                                               random_color_jitter=False, resize_size=args.resize_size,
                                               norm_mean=args.norm_mean, norm_std=args.norm_std,
                                               auto_augment=args.auto_augment)
    train_target_transform = MultipleApply([weak_augment, strong_augment])
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)

    print("train_source_transform: ", train_source_transform)
    print("train_target_transform: ", train_target_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_labeled_target_dataset, train_unlabeled_target_dataset, val_dataset, test_dataset, \
    num_classes, args.class_names = utils.get_semi_dataset(args.data, args.root, args.source, args.target, args.shots,
                                                      train_source_transform, val_transform,
                                                      train_target_transform=train_target_transform)

    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)

    train_labeled_target_loader = DataLoader(train_labeled_target_dataset,
                                             batch_size=min(args.unlabeled_batch_size, num_classes),
                                             shuffle=True, num_workers=args.workers, drop_last=True)

    train_unlabeled_target_loader = DataLoader(train_unlabeled_target_dataset, batch_size=args.unlabeled_batch_size,
                                               shuffle=True, num_workers=args.workers, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_unlabeled_target_iter = ForeverDataIterator(train_unlabeled_target_loader)
    train_labeled_target_iter = ForeverDataIterator(train_labeled_target_loader)

    # Create Model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    print(classifier)

    # Define Optimizer and Lr Scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                    nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # Instantiate Nuclear Wasserstein Distance(NWD)
    discrepancy = NuclearWassersteinDiscrepancy(classifier.head).to(device)

    # Resume From the Best Checkpoint(If not in training mode)
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # Analysis the Model if in Analysis Mode
    if args.phase == 'analysis':
        # Extract Features from Both Domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_unlabeled_target_loader, feature_extractor, device)

        # Plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)

        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # Start Training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr())

        # Train for One Epoch
        train(train_source_iter, train_unlabeled_target_iter, train_labeled_target_iter, classifier,
              discrepancy, optimizer, lr_scheduler, epoch, args)

        # Evaluate on Validation Dataset
        acc1 = utils.validate(val_loader, classifier, args, device)

        # Record The Best Acc@1 and Save Checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # Evaluate on Testdata
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_unlabeled_target_iter: ForeverDataIterator,
          train_labeled_target_iter: ForeverDataIterator, model: ImageClassifier, domain_discrepancy,
          optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')  # Training Time for Current Batch
    data_time = AverageMeter('Data', ':5.2f')  # Data Loading Time for Current Batch
    cls_losses = AverageMeter('Cls loss', ':6.2f')  # Classification Loss
    self_training_losses = AverageMeter('Self Training Loss', ':6.2f')  # Self Training Loss
    discre = AverageMeter('Discrepancy Loss', ':6.2f')  # Domain Discrepancy Loss
    losses = AverageMeter('Loss', ':6.2f')  # Total Loss
    cls_accs = AverageMeter('Cls Acc', ':3.1f')  # Classification Accuracy
    pseudo_label_ratios = AverageMeter('Pseudo Label Ratio', ':3.1f')  # Ratios of Pseudo Labels in Target Domain
    pseudo_label_accs = AverageMeter('Pseudo Label Acc', ':3.1f')  # Accuracy of Pseudo Labels in Target Domain

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, self_training_losses, losses, cls_accs, pseudo_label_ratios,
         pseudo_label_accs, discre],
        prefix="Epoch: [{}]".format(epoch)
    )

    self_training_criterion = ConfidenceBasedSelfTrainingLoss(args.threshold).to(device)

    # Switch to Mode Train
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        (x_t_u, x_t_u_strong), labels_t_u = next(train_unlabeled_target_iter)[:2]
        # TEST____________________________________________________
        (x_t_l, _), labels_t_l = next(train_labeled_target_iter)[:2]

        x_s = x_s.to(device)
        x_t_u = x_t_u.to(device)
        x_t_u_strong = x_t_u_strong.to(device)
        labels_s = labels_s.to(device)
        labels_t_u = labels_t_u.to(device)

        # TEST________________________________________________________
        x_t_l = x_t_l.to(device)
        labels_t_l = labels_t_l.to(device)

        # Measure Data Loading Time
        data_time.update(time.time() - end)

        # Clear Grad
        optimizer.zero_grad()

        # Cross Entropy Loss over Source Data
        y_s, f_s = model(x_s)
        cls_loss = F.cross_entropy(y_s, labels_s)

        # Cross Entropy Loss over labeled Target Data, TEST______________________________________
        y_t_l, _ = model(x_t_l)
        cls_loss_t_l = F.cross_entropy(y_t_l, labels_t_l)
        cls_loss_t_l.backward()

        # Nuclear Discrepancy Loss over Source and Target
        _, f_t_u = model(x_t_u)

        f = torch.cat([f_s, f_t_u], dim=0)
        discrepancy_loss = -domain_discrepancy(f)

        # Compute Output upon x_t_u
        with torch.no_grad():
            y_t_u, _ = model(x_t_u)

        # Self-training Loss
        y_t_u_strong, _ = model(x_t_u_strong)
        self_training_loss, mask, pseudo_labels = self_training_criterion(y_t_u_strong, y_t_u)

        # Total loss
        total_loss = cls_loss + args.trade_off_disc * discrepancy_loss + args.trade_off_self * self_training_loss

        total_loss.backward()

        # Measure Accuracy and Record Loss
        losses.update(total_loss.item(), x_s.size(0))
        cls_losses.update(cls_loss.item(), x_s.size(0))
        self_training_losses.update(self_training_loss.item(), x_s.size(0))
        discre.update(discrepancy_loss.item(), x_s.size(0))

        cls_acc = accuracy(y_s, labels_s)[0]
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # Ratio of Pseudo Labels
        n_pseudo_labels = mask.sum()
        ratio = n_pseudo_labels / x_t_u.size(0)
        pseudo_label_ratios.update(ratio.item() * 100, x_t_u.size(0))

        # Accuracy of Pseudo Labels
        if n_pseudo_labels > 0:
            pseudo_labels = pseudo_labels * mask - (1 - mask)
            n_correct = (pseudo_labels == labels_t_u).float().sum()
            pseudo_labels_acc = n_correct / n_pseudo_labels * 100
            pseudo_label_accs.update(pseudo_labels_acc.item(), n_pseudo_labels)

        # Compute Gradients and DO SGD
        optimizer.step()
        lr_scheduler.step()

        # Measure Elapsed Time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FixMatch for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.5, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.5 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    parser.add_argument('--auto-augment', default='rand-m10-n2-mstd2', type=str,
                        help='AutoAugment policy (default: rand-m10-n2-mstd2)')
    parser.add_argument('--shots', default=1, type=int,
                        help='Shots for labeled target data, 1 or 3')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off-disc', default=1., type=float,
                        help='the trade-off hyper-parameter for discrepancy loss')
    parser.add_argument('--trade-off-self', default=1., type=float,
                        help='the trade-off hyper-parameter for self-training loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('-ub', '--unlabeled-batch-size', default=32, type=int,
                        help='mini-batch size of unlabeled data (target domain) (default: 32)')
    parser.add_argument('--threshold', default=0.9, type=float,
                        help='confidence threshold')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0004, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='fixmatch',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
