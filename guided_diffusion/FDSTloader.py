# coding=utf-8
'''创建DatasetFromFolder类，处理数据集'''
from os import listdir
from os.path import join
import types
import torch
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import numpy as np
# from util import is_image_file, load_img#, load_gt
from pylab import *
from torch.autograd import Variable
import os
import cv2
import h5py
import pdb
import glob


# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
class FDSTDataset(data.Dataset):
    def __init__(self, image_dir, transform_list, test_flag=False):
        super(FDSTDataset, self).__init__()
        self.photo_path = image_dir  # join(image_dir, "new_gt")
        # self.den_path = join(image_dir, "b")
        self.test_flag = test_flag
        # self.image_filenames = [x for x in listdir(self.photo_path) if is_image_file(x)]
        self.path_sets = [os.path.join(self.photo_path, x) for x in listdir(self.photo_path) if
                          os.path.isdir(os.path.join(self.photo_path, x))]
        self.image_filenames = []
        self.transform = transform_list
        for path in self.path_sets:
            for img_path in glob.glob(os.path.join(path, '*.jpg')):
                self.image_filenames.append(img_path)  # 将jpg路径名称添加到列表中

        if test_flag:
            self.image_filenames = sorted(self.image_filenames)  # 排序？
        # transform_list = [transforms.ToTensor(),
        #                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])]#PILimage转换为Tensor,并进行归一化，三个通道分别对应三组均值+标准差
        # self.transform = transforms.Compose(transform_list)  # 对图片进行ToTensor,normalize的转换
        # self.mean = torch.tensor([0.37653536, 0.37653536, 0.37653536]).view(3, 1, 1)
        # self.std = torch.tensor([0.20983484, 0.20983484, 0.20983484]).view(3, 1, 1)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # 普通向量转化为3维向量对应三个通道？
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        # self.imgnum = imgnum
        self.img_pool = {}  # 字典
        # pdb.set_trace()
        self.get_all_img()

    '''单张输入getitem'''
    def __getitem__(self, index):
        filepath = self.image_filenames[index]
        if not self.test_flag:
            img, den = self.img_pool[self.image_filenames[index]]
            img = Image.open(filepath)
            img = self.transform(img)

            return img, den

        else:
            img, den = self.img_pool[self.image_filenames[index]]
            img = Image.open(filepath)
            img = self.transform(img)
            count = np.sum(den)/5
            return img, count, self.image_filenames[index]

    '''多张输入getitem'''
    # def __getitem__(self, index):  # 获取当前index对应的图片以及之前T（imgnum)帧的完全处理好的所有img和den
    #     # Load Image
    #     filepath = join(self.photo_path, self.image_filenames[index])
    #     if not self.istest:
    #         input, target = load_img(self.img_pool, self.image_filenames[index], self.imgnum,
    #                                  self.istest)  # 得到处理后的图片以及密度图
    #         input = torch.from_numpy(input.copy()).type(
    #             torch.FloatTensor)  # copy与源数组分割，更改源数组不会对新数组产生影响;from_numpy将数组转换为张量且共享内存，使张量和数组能够同步改变；type
    #         # :按照Floattensor类型进行强制转换并返回
    #
    #         c, h, w = input.shape
    #         input = input.view((self.imgnum, 3, h, w))  # 将拼接后的图片再分成imgnum个维度？
    #         input = (input / 255 - self.mean) / self.std
    #         input = input.permute(1, 0, 2, 3)  # tensor维度换位为c,imgnum,h,w
    #         # input = input.view((c, h, w))
    #         den = torch.from_numpy(target.copy()).type(torch.FloatTensor).unsqueeze(0)  # 增加一个维度，将c,h,w变为1,c,h,w
    #         # oi = torch.from_numpy(roi.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    #
    #         return input, den
    #         # return torch.cat((input1, input2, input3),0), den
    #     else:
    #         # input,input2, input3, count = load_img(join(self.photo_path, self.image_filenames[index]),self.istest)
    #         input, count, filename = load_img(self.img_pool, self.image_filenames[index], self.imgnum, self.istest)
    #         input = torch.from_numpy(input.copy()).type(torch.FloatTensor)
    #         c, h, w = input.shape
    #         input = input.view((self.imgnum, 3, h, w))
    #         input = (input / 255 - self.mean) / self.std  # 标准化，输入到预加载模型之前需要对其进行固定标准化?
    #         input = input.permute(1, 0, 2, 3)
    #         # roi = torch.from_numpy(roi.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    #         return input, torch.from_numpy(
    #             count), filename  # , roi, torch.from_numpy(roi.copy()).type(torch.FloatTensor).unsqueeze(0)

    def __len__(self):
        return len(self.image_filenames)

    def get_all_img(self):  # 获取包含所有filename以及图片+密度图的字典
        for filename in self.image_filenames:
            filepath = filename
            # pdb.set_trace()
            # img = cv2.imread(filepath)[:, :, ::-1]  # cv2.read返回(h,w,c),由于imread返回的通道顺序为BGR，所以需要转换为RGB方便处理
            img = Image.open(filepath)
            denpath = filepath.replace('.jpg', '.h5')
            denfile = h5py.File(denpath, 'r')
            # den = np.asarray(denfile['density'])*5#总数增加五倍
            # img = np.asarray(denfile['image'])
            den = np.asarray(denfile['density']) * 5
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            den = transform(den)
            den = den.permute(0, 2, 1)
            # pdb.set_trace() den = cv2.resize(den,(den.shape[1]//2,den.shape[0]//2),interpolation =
            # cv2.INTER_CUBIC)*4#宽高缩小为原来一半，总数仍然是五倍
            self.img_pool[filename] = (img, den)  # 将字典中的filename对应到img和den
            # pdb.set_trace()
        # roi = h5py.File("/home/smj/dataset/UCSD/roi.h5", 'r')
        # roi = np.asarray(roi['/roi'])
        '''
        for filename in self.img_all_name:
            filepath = join(self.all_path, filename)
            img = cv2.imread(filepath)[:,:, ::-1]
            self.all_img[filename] = img
        '''


def load_img(img_pool, filename, imgnum, istest):
    img, den = img_pool[filename]
    path, index = filename[:-7], int(filename[-7:-4])
    deltaT = 1
    '''前后共七张'''
    for i in range(1, int(imgnum / 2) + 1):  # 取当前帧的前img_num/2帧
        filepath_last = path + "%03d.jpg" % (index - i * deltaT)
        if filepath_last in img_pool:
            img_last, den_last = img_pool[filepath_last]
        else:
            img_last, den_last = img_pool[path + "%03d.jpg" % (1)]
        img = np.concatenate((img, img_last), axis=2)  # 在深度上拼接
        if i == 2:
            den = np.dstack((den, den_last))  # 沿深度拼接为三维数组
    for i in range(1, int(imgnum / 2) + 1):  # 取当前帧的后img_num/2帧
        filepath_last = path + "%03d.jpg" % (index + i * deltaT)
        if filepath_last in img_pool:
            img_first, den_first = img_pool[filepath_last]
        else:
            img_first, den_first = img_pool[path + "%03d.jpg" % (150)]
        img = np.concatenate((img_first, img), axis=2)  # 在深度上拼接
        if i == 2:
            den = np.dstack((den_first, den))  # 沿深度拼接为三维数组
    h, w, c = img.shape
    isgray = False
    if not istest:
        new_w = 512
        new_h = 512
        # new_w = 1920
        # new_h = 1080
        # pdb.set_trace()
        x = random.randint(0, w - new_w)  # //8 * 8
        y = random.randint(0, h - new_h)  # //8  * 8

        img = img[y:y + new_h, x:x + new_w, :]  # 对图片进行随机裁切
        den = den[y // 8:y // 8 + new_h // 8, x // 8:x // 8 + new_w // 8, :]  # 将密度图裁剪成(new_h/8,new_w/8)大小
        return img.transpose((2, 0, 1)), den.transpose((2, 0, 1))  # , roi.transpose((2,0,1));将数组进行维度换位为(c,h,w)

    else:
        # img = img.resize((1920, 1080),Image.BICUBIC)
        # img_last = img_last.resize((1920, 1080),Image.BICUBIC)
        # img_next = img_next.resize((1920, 1080),Image.BICUBIC)

        # img = img.resize((img.size[0]//8*8, img.size[1]//8*8),Image.BICUBIC)
        # img_last = img_last.resize((img_last.size[0]//8*8, img_last.size[1]//8*8),Image.BICUBIC)
        # img_next = img_next.resize((img_next.size[0]//8*8, img_next.size[1]//8*8),Image.BICUBIC)
        # img = cv2.resize(img,(156, 236),interpolation = cv2.INTER_CUBIC)
        # roi = cv2.resize(roi,(232,152),interpolation = cv2.INTER_NEAREST)
        gt_cnt = []
        for i in range(int(imgnum / 2)):
            gt_cnt += [np.sum(den[:, :, i]) / 5]
        # pdb.set_trace()
        return img.transpose((2, 0, 1)), np.array(gt_cnt), filename


class CustSamp(Sampler):
    def __init__(self, dataset):
        # num = int(len(dataset)/4)
        num = int(len(dataset))
        self.first_half_indices = list(range(2, num, 6)) + list(range(3, num, 6))
        self.first_half_indices.sort()
        # self.first_half_indices = list(range(7,num,8))

    def __iter__(self):
        return iter(self.first_half_indices)

    def __len__(self):
        return len(self.first_half_indices)
