from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class Office31_semi(ImageList):

    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/d9bca681c71249f19da2/?dl=1"),
        ("amazon", "amazon.tgz", "https://cloud.tsinghua.edu.cn/f/edc8d1bba1c740dc821c/?dl=1"),
        ("dslr", "dslr.tgz", "https://cloud.tsinghua.edu.cn/f/ca6df562b7e64850ad7f/?dl=1"),
        ("webcam", "webcam.tgz", "https://cloud.tsinghua.edu.cn/f/82b24ed2e08f4a3c8888/?dl=1"),
    ]
    image_list = {
        "A": "image_list/amazon.txt",
        "D": "image_list/dslr.txt",
        "W": "image_list/webcam.txt",
        "A_target_labeled_1": "image_list/semi/labeled_target_images_amazon_1.txt",
        "A_target_labeled_3": "image_list/semi/labeled_target_images_amazon_3.txt",
        "A_target_unlabeled_1": "image_list/semi/unlabeled_target_images_amazon_1.txt",
        "A_target_unlabeled_3": "image_list/semi/unlabeled_target_images_amazon_3.txt",
        "D_target_labeled_1": "image_list/semi/labeled_target_images_dslr_1.txt",
        "D_target_labeled_3": "image_list/semi/labeled_target_images_dslr_3.txt",
        "D_target_unlabeled_1": "image_list/semi/unlabeled_target_images_dslr_1.txt",
        "D_target_unlabeled_3": "image_list/semi/unlabeled_target_images_dslr_3.txt",
        "W_target_labeled_1": "image_list/semi/labeled_target_images_webcam_1.txt",
        "W_target_labeled_3": "image_list/semi/labeled_target_images_webcam_3.txt",
        "W_target_unlabeled_1": "image_list/semi/unlabeled_target_images_webcam_1.txt",
        "W_target_unlabeled_3": "image_list/semi/unlabeled_target_images_webcam_3.txt"
    }

    CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
               'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
               'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
               'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(Office31_semi, self).__init__(root, Office31_semi.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())