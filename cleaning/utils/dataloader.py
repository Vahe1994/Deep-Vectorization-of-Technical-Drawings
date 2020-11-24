import os
import numpy as np
import torch
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import PIL


class MakeData(Dataset):
    def __init__(self, img_path, img_y_path, transform=None, tr=None):
        tmp_df_x = []
        tmp_df_y = []
        for it in os.listdir(img_path):
            if '_' not in it and '.svg' not in it:
                tmp_df_x.append(it)
        tmp_df_x.sort()
        for it in tmp_df_x:
            tmp_df_y.append(it[:-4] + '_gt.png')
        self.img_path_x = img_path
        self.img_path_y = img_y_path
        self.transform = transform
        self.tr = tr
        self.X_train = tmp_df_x
        self.y_train = tmp_df_y

        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def randomCrop(self, img, mask, width, height):
        print(img.size, width, height)
        assert img.size[0] >= height
        assert img.size[1] >= width
        assert img.size[0] == mask.size[0]
        assert img.size[1] == mask.size[1]
        x = random.randint(0, img.size[1] - width)
        y = random.randint(0, img.size[0] - height)
        img = img.crop((y, x, y + height, x + width))
        mask = mask.crop((y, x, y + height, x + width))
        return img, mask

    def transformation(self, img, img_y):
        img = PIL.ImageOps.invert(img)
        img_y = PIL.ImageOps.invert(img_y)
        if (np.random.uniform() < 0.5):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_y = img_y.transpose(Image.FLIP_LEFT_RIGHT)
        elif (np.random.uniform() < 0.5):
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_y = img_y.transpose(Image.FLIP_TOP_BOTTOM)
        rn = np.random.uniform(0, 270)
        img = img.rotate(rn)
        img_y = img_y.rotate(rn)
        mini = 512
        img, img_y = self.randomCrop(img, img_y, mini, mini)
        #         img_y = np.array(img_y)
        #         a=(img_y>0).astype(int)
        img = PIL.ImageOps.invert(img)
        img_y = PIL.ImageOps.invert(img_y)
        return img, img_y

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path_x, self.X_train[index]))
        img = img.convert('L')
        img_y = Image.open(os.path.join(self.img_path_y, self.y_train[index]))
        img_y = img_y.convert('L')

        if self.tr is not None:
            img, img_y = self.transformation(img, img_y)
            img = self.transform(img)
        else:
            # change back
            img_y = img_y.resize((800, 800))
            img = img.resize(img_y.size)
            img = self.trans(img)
        img_y = np.array(img_y).astype(float)
        img_y = img_y / 255.0
        #         img_y = (img_y>0).astype(int)
        label = torch.from_numpy(img_y)
        return img, label

    def __len__(self):
        return len(self.X_train)


class MakeDataSynt(Dataset):
    def __init__(self, img_path, img_y_path, transform=None, tr=None):
        tmp_df_x = []
        tmp_df_y_h = []
        tmp_df_y_nh = []

        for it in os.listdir(img_path):
            if '_' not in it and '.svg' not in it:
                tmp_df_x.append(it)
        tmp_df_x.sort()

        for it in tmp_df_x:
            tmp_df_y_h.append(it[:-4] + '_h_gt.png')

        for it in tmp_df_x:
            tmp_df_y_nh.append(it[:-4] + '_nh_gt.png')

        self.img_path_x = img_path
        self.img_path_y_h = img_y_path
        self.img_path_y_nh = img_y_path

        self.transform = transform
        self.tr = tr
        self.X_train = tmp_df_x
        self.y_train_h = tmp_df_y_h
        self.y_train_nh = tmp_df_y_nh

        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def randomCrop(self, img, mask, mask_1, width, height):
        assert img.size[0] >= height
        assert img.size[1] >= width
        assert img.size[0] == mask.size[0]
        assert img.size[1] == mask.size[1]
        x = random.randint(0, img.size[1] - width)
        y = random.randint(0, img.size[0] - height)
        img = img.crop((y, x, y + height, x + width))
        mask = mask.crop((y, x, y + height, x + width))
        mask_1 = mask_1.crop((y, x, y + height, x + width))
        return img, mask, mask_1

    def transformation(self, img, img_y, img_y_nh):
        img = PIL.ImageOps.invert(img)
        img_y = PIL.ImageOps.invert(img_y)
        img_y_nh = PIL.ImageOps.invert(img_y_nh)
        if (np.random.uniform() < 0.5):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_y = img_y.transpose(Image.FLIP_LEFT_RIGHT)
            img_y_nh = img_y_nh.transpose(Image.FLIP_LEFT_RIGHT)
        elif (np.random.uniform() < 0.5):
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_y = img_y.transpose(Image.FLIP_TOP_BOTTOM)
            img_y_nh = img_y_nh.transpose(Image.FLIP_TOP_BOTTOM)
        rn = np.random.uniform(0, 270)
        img = img.rotate(rn)
        img_y = img_y.rotate(rn)
        img_y_nh = img_y_nh.rotate(rn)
        mini = 512
        img, img_y, img_y_nh = self.randomCrop(img, img_y, img_y_nh, mini, mini)
        #         img_y = np.array(img_y)
        #         img_y = np.array(img_y)
        #         a=(img_y>0).astype(int)
        img = PIL.ImageOps.invert(img)
        img_y = PIL.ImageOps.invert(img_y)
        img_y_nh = PIL.ImageOps.invert(img_y_nh)
        return img, img_y, img_y_nh

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path_x, self.X_train[index]))
        img = img.convert('RGB')
        img_y_h = Image.open(os.path.join(self.img_path_y_h, self.y_train_h[index]))
        img_y_h = img_y_h.convert('L')

        img_y_nh = Image.open(os.path.join(self.img_path_y_nh, self.y_train_nh[index]))
        img_y_nh = img_y_nh.convert('L')
        if self.tr is not None:
            img, img_y_h, img_y_nh = self.transformation(img, img_y_h, img_y_nh)
            img = self.transform(img)
        else:
            # img_y = img_y_h.resize((800, 800))
            # img = img.resize(img_y_h.size)
            img = self.trans(img)
        #         img = self.transform(img)

        img_y_h = np.array(img_y_h).astype(float)
        img_y_h = img_y_h / 255.0

        img_y_nh = np.array(img_y_nh).astype(float)
        img_y_nh = img_y_nh / 255.0

        label_h = torch.from_numpy(img_y_h)
        label_nh = torch.from_numpy(img_y_nh)

        if self.tr is not None:
            return img, label_h, label_nh
        else:
            img_t = torch.ones(img.shape[0], img.shape[1] + (32 - img.shape[1] % 32),
                               img.shape[2] + (32 - img.shape[2] % 32))
            img_t[:, :img.shape[1], :img.shape[2]] = img

            label_h_t = torch.ones(label_h.shape[0] + (32 - label_h.shape[0] % 32),
                                   label_h.shape[1] + (32 - label_h.shape[1] % 32))
            label_h_t[:label_h.shape[0], :label_h.shape[1]] = label_h

            label_nh_t = torch.ones(label_nh.shape[0] + (32 - label_nh.shape[0] % 32),
                                    label_nh.shape[1] + (32 - label_nh.shape[1] % 32))
            label_nh_t[:label_nh.shape[0], :label_nh.shape[1]] = label_nh

            return img_t, label_h_t, label_nh_t

    def __len__(self):
        return len(self.X_train)


class MakeDataVectorField(Dataset):
    def __init__(self, data_path):
        df = []
        for name in os.listdir(data_path):
            if name.endswith('.npy'):
                df.append(name)

        df.sort()
        self.data_path = data_path
        self.data_files = df

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):

        with open(self.data_path + self.data_files[index], 'rb') as inp:
            img_field = np.load(inp)

        img_field = torch.from_numpy(img_field)
        img = img_field[0]
        img /= 255
        label = img_field[1:]

        return img, label