from PIL import Image
from torch.utils.data import Dataset
import random
import util as Util
import numpy as np
import cv2
from matplotlib import pyplot as plt
# from model.utils import categories
def color_filter(image):
    gray_SR = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_SR_gray = np.zeros_like(image)
    img_SR_gray[:, :, 0] = gray_SR
    img_SR_gray[:, :, 1] = gray_SR
    img_SR_gray[:, :, 2] = gray_SR

    result = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 0, 20])
    upper1 = np.array([40, 255, 255])

    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([125, 0, 20])
    upper2 = np.array([179, 255, 255])

    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)

    full_mask = lower_mask + upper_mask
    # print(np.unique(full_mask))
    neg_full_mask = 255 - full_mask

    result = cv2.bitwise_and(result, result, mask=full_mask)
    result_neg = cv2.bitwise_and(img_SR_gray, img_SR_gray, mask=neg_full_mask)
    return result_neg + result[:, :, ::-1]

class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, split='train', data_len=-1, image_size=512):
        self.datatype = datatype
        self.data_len = data_len
        self.split = split
        self.image_size = image_size


        if datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/{}'.format(dataroot, 'image'))
            self.hr_path = Util.get_paths_from_images(
                '{}/{}'.format(dataroot, 'label'))
            # self.style_path = Util.get_paths_from_images(
            #     '{}/hr_{}_style'.format(dataroot, r_resolution))
            # if self.need_LR:
            #     self.lr_path = Util.get_paths_from_images(
            #         '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # index_style = np.random.randint(0, self.data_len)

        # img_HR = Image.open(self.hr_path[index]).convert("RGB").resize((128, 128))
        # img_SR = Image.open(self.sr_path[index]).convert("RGB").resize((128, 128))
        # img_style = Image.open(self.style_path[index]).convert("RGB").resize((64, 64))

        img_HR = Image.open(self.hr_path[index]).convert("RGB").resize((self.image_size, self.image_size))
        img_SR = Image.open(self.sr_path[index]).convert("RGB").resize((self.image_size, self.image_size))
        # img_style = Image.open(self.style_path[index]).convert("RGB")
        # img_style = Image.open(self.style_path[index_style]).convert("RGB")

        # img_HR = cv2.imread(self.hr_path[index])
        # img_HR = img_HR[:, :, ::-1]
        # img_HR = img_HR.copy()
        # img_SR = cv2.imread(self.sr_path[index])
        # img_style = cv2.imread(self.style_path[index])
        # img_style = img_style[:, :, ::-1]
        # img_style = img_style.copy()
        # img_SR = color_filter(img_SR)
        # img_SR = img_SR.copy()
        # img_style = Image.open(self.sr_path[index]).convert("RGB")
        [img_SR, img_HR] = Util.transform_augment(
            [img_SR, img_HR], split=self.split, min_max=(-1, 1))
        return {'high': img_HR, 'low': img_SR}
