import os
import cv2
import numpy as np
import random
import torch
from torch.nn import functional as F
from torch.utils import data
from PIL import Image


class RoadBase(data.Dataset):
    def __init__(self, ignore_label=255, base_size=1024, crop_size=(512, 512),
                 downsample_rate=1, scale_factor=16,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1.0 / downsample_rate
        self.files = []
        self.num_classes = 2

    def __len__(self):
        return len(self.files)

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            if image.ndim == 3:
                pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                               cv2.BORDER_CONSTANT, value=padvalue)
            else:
                pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                               cv2.BORDER_CONSTANT, value=(padvalue,))
        return pad_image

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))
        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        return image, label

    def multi_scale_aug(self, image, label=None, rand_scale=1, rand_crop=True):
        long_size = int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = int(h * long_size / w + 0.5)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            return image
        if rand_crop:
            image, label = self.rand_crop(image, label)
        return image, label

    def gen_sample(self, image, label, multi_scale=True, is_flip=True):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)

        image = self.input_transform(image)
        label = label.astype(np.int64)
        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        return image, label

    def inference(self, config, model, image, flip=False):
        size = image.size()
        pred = model(image)
        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]
        pred = F.interpolate(input=pred, size=size[-2:],
                             mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            if config.MODEL.NUM_OUTPUTS > 1:
                flip_output = flip_output[config.TEST.OUTPUT_INDEX]
            flip_output = F.interpolate(input=flip_output, size=size[-2:],
                                        mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred


class DeepglobeRoad(RoadBase):
    def __init__(self, root, list_path, num_samples=None, num_classes=2,
                 multi_scale=True, flip=True, ignore_label=255,
                 base_size=1024, crop_size=(512, 512), downsample_rate=1,
                 scale_factor=16, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super(DeepglobeRoad, self).__init__(ignore_label, base_size, crop_size,
                                            downsample_rate, scale_factor, mean, std)
        self.root = root
        self.list_path = list_path
        self.multi_scale = multi_scale
        self.flip = flip
        self.class_weights = torch.FloatTensor([0.05, 0.95]).cuda()

        with open(os.path.join(root, list_path)) as f:
            self.img_list = [line.strip().split() for line in f]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            if len(item) >= 2:
                img_path, label_path = item[0], item[1]
            else:
                continue
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({"img": img_path, "label": label_path, "name": name})
        return files

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, item["img"]), cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(self.root, item["label"]), cv2.IMREAD_GRAYSCALE)
        label[label == 255] = 1
        label[label != 1] = 0
        size = image.shape

        if 'val' in self.list_path or 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), label.copy(), np.array(size), name

        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)
        return image.copy(), label.copy(), np.array(size), name


class RoadDataset2(RoadBase):
    def __init__(self, root, list_path, num_samples=None, num_classes=2,
                 multi_scale=True, flip=True, ignore_label=255,
                 base_size=1024, crop_size=(512, 512), downsample_rate=1,
                 scale_factor=16, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super(RoadDataset2, self).__init__(ignore_label, base_size, crop_size,
                                           downsample_rate, scale_factor, mean, std)
        self.root = root
        self.list_path = list_path
        self.multi_scale = multi_scale
        self.flip = flip
        self.class_weights = torch.FloatTensor([0.05, 0.95]).cuda()

        with open(os.path.join(root, list_path)) as f:
            self.img_list = [line.strip().split() for line in f if line.strip()]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            if len(item) >= 2:
                img_path, label_path = item[0], item[1]
            else:
                continue
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({"img": img_path, "label": label_path, "name": name})
        return files

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, item["img"]), cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(self.root, item["label"]), cv2.IMREAD_GRAYSCALE)
        label[label == 255] = 1
        label[label != 1] = 0
        size = image.shape

        if 'val' in self.list_path or 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), label.copy(), np.array(size), name

        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)
        return image.copy(), label.copy(), np.array(size), name
