# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset
import torchvision.transforms as T
from fastreid.data.transforms import ToTensor
from PIL import Image
import numpy as np
import cv2

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.mask_transform = None
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

        tmp = self.__getitem__(0)
        img = tmp['images']
        _, rH, rW = img.shape  # 256, 128
        # print("common, init, what are rH and rW", rH, rW)
        self.mask_transform = T.Compose([T.Resize([rH, rW], interpolation=3),
                                         ToTensor()])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = read_image(img_path)  # PIL (W, H)
        W, H = img.size
        # print("common>CommDataset>getitem: W, H", W, H)
        if self.transform is not None: img = self.transform(img)  # torch.Size([3, 256, 128])
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]

        mask = np.zeros([H, W], dtype=np.uint8)
        try:
            mask_path = img_item[4]
            with open(mask_path, 'r') as f:
                x, y, w, h = f.read().strip().split(',')
                x, y, w, h = int(x), int(y), int(w), int(h)
                x1 = x if x > 0 else 0
                y1 = y if y > 0 else 0
                x2 = (x + w) if (x + w) <= W else W
                y2 = (y + h)*2 if (y + h)*2 <= H else H
                # y2 = H
                mask[y1:y2, x1:x2] = 1
                # print("common>getitem>mask before transform", mask[y1:y2, x1:x2])
        except:
            pass
        mask = Image.fromarray(mask)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        # print("=====================================")
        # show_mask = np.array(mask)
        # print("common>getitem>mask after transform", show_mask[y1:y2, x1:x2])
        # RGB 图像转换成为 array 格式
        gray = np.asarray(img)
        # 转换成灰度图
        # print("common>>> gray array size", gray.shape)
        gray = np.transpose(gray, (1, 2, 0))
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        # 转换为二值图
        _, norm_gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        norm_gray = gray / 255.0  # 范围变成[0,1]
        norm_gray = Image.fromarray(norm_gray)
        if self.mask_transform is not None:
            norm_gray = self.mask_transform(norm_gray)

        norm_gray = None
        # region sketch 的轮廓图
        sketch = cv2.imread(img_item[3], cv2.IMREAD_GRAYSCALE)
        sketch = sketch.astype(np.float32)
        # sketch = sketch / np.max(sketch)
        sketch = sketch / 255.0  # 两种方式差不多
        # cv2.imshow("show",sketch)
        # cv2.waitKey()
        sketch = Image.fromarray(sketch)
        if self.mask_transform is not None:
            sketch = self.mask_transform(sketch)
            # sketch = sketch[:, :, 0]
        # endregion

        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "sketch": sketch,
            "masks": mask,
            "gray": norm_gray
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
