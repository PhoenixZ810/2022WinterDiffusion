# coding=utf-8
'''两个函数
is_image_file返回所有带png/jpg/jpeg后缀的文件名
load_img获取当前图片及之前T（imgnum)帧全部拼接在一起并裁剪后的图片以及密度图
cal_keypoints
cal_innner_area
save_img存储图片数组并返回某一通道的平均值（人数）？
'''


import numpy as np
from PIL import Image
from imageio import imsave
import random
import math
import matplotlib.pyplot as plt
import h5py
import cv2
from scipy.ndimage import filters
import scipy.io as sio
import copy
import os
import time
import pdb
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
def is_image_file(filename):
    return  any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(img_pool, filename, imgnum, istest):
    img, den = img_pool[filename]
    path, index = filename[:-7], int(filename[-7:-4])
    deltaT = 1
    # if not istest and random.random()>0.5:
    #    deltaT = 2
    # '''前八张'''
    # for i in range(1, imgnum):#取当前帧的前img_num帧
    #     filepath_last = path + "%03d.jpg" % (index - i * deltaT)
    #     if filepath_last in img_pool:
    #         img_last, den_last = img_pool[filepath_last]
    #     else:
    #         img_last, den_last = img_pool[path + "%03d.jpg" % (1)]
    #     img = np.concatenate((img, img_last), axis=2)  # 在深度上拼接
    #     den = np.dstack((den, den_last))  # 沿深度拼接为三维数组
    '''前后共七张'''
    for i in range(1, int(imgnum/2)+1):#取当前帧的前img_num/2帧
        filepath_last = path + "%03d.jpg" % (index - i * deltaT)
        if filepath_last in img_pool:
            img_last, den_last = img_pool[filepath_last]
        else:
            img_last, den_last = img_pool[path + "%03d.jpg" % (1)]
        img = np.concatenate((img, img_last), axis=2)  # 在深度上拼接
        if i == 2:
            den = np.dstack((den, den_last))  # 沿深度拼接为三维数组
    for i in range(1, int(imgnum/2)+1):#取当前帧的后img_num/2帧
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
        den = den[y // 8:y // 8 + new_h // 8, x // 8:x // 8 + new_w // 8, :]#将密度图裁剪成(new_h/8,new_w/8)大小
        return img.transpose((2, 0, 1)), den.transpose((2, 0, 1)) # , roi.transpose((2,0,1));将数组进行维度换位为(c,h,w)

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
        for i in range(int(imgnum/2)):
            gt_cnt += [np.sum(den[:, :, i]) / 5]
        # pdb.set_trace()
        return img.transpose((2, 0, 1)), np.array(gt_cnt), filename

'''#FDST load_img'''
'''
def load_img(filepath, imgnum, istest=False):
    img = cv2.imread(filepath)[:,:, ::-1]
    denpath = filepath.replace('.jpg','.h5')
    den = h5py.File(denpath, 'r')
    den = np.asarray(den['density']) * 100
    den = cv2.resize(den,(den.shape[1]//2,den.shape[0]//2),interpolation = cv2.INTER_CUBIC)*4
    path, index = filepath[:-7], int(filepath[-7:-4])
    deltaT = 1
    if not istest and random.random()>0.5:
        deltaT = 2
    for i in range(1, imgnum):
        filepath_last = path + "%03d.jpg"%(index-i*deltaT)
        if os.path.exists(filepath_last):
            img_last = cv2.imread(filepath_last)[:,:, ::-1]
            denpath_last = filepath_last.replace('.jpg','.h5')
            den_last = h5py.File(denpath_last, 'r')
        else:
            img_last = cv2.imread(path + "%03d.jpg"%(1))[:,:, ::-1]
            den_last = h5py.File(path + "%03d.h5"%(1), 'r')
        den_last = np.asarray(den_last['density']) * 100
        den_last = cv2.resize(den_last,(den_last.shape[1]//2,den_last.shape[0]//2),interpolation = cv2.INTER_CUBIC)*4
        img = np.concatenate((img, img_last), axis=2)
        den = np.dstack((den, den_last))

    h, w, c = img.shape
    isgray = False

    if not istest:
        new_w = 512 
        new_h = 512
        x = random.randint(0,w-new_w)# //8 * 8
        y = random.randint(0,h-new_h)# //8  * 8
        img = img[y:y+new_h, x:x + new_w, :]
        den = den[y//8:y//8+new_h//8, x//8:x//8 + new_w//8,:]

        if random.random()>0.5:
            return img.transpose((2,0,1)), den.transpose((2,0,1))
        else:
            den = den[:, ::-1, :]
            img = img[:, ::-1, :]
            return img.transpose((2,0,1)), den.transpose((2,0,1))
    else:
        
        #img = img.resize((1920, 1080),Image.BICUBIC)
        #img_last = img_last.resize((1920, 1080),Image.BICUBIC)
        #img_next = img_next.resize((1920, 1080),Image.BICUBIC)
        
        #img = img.resize((img.size[0]//8*8, img.size[1]//8*8),Image.BICUBIC)
        #img_last = img_last.resize((img_last.size[0]//8*8, img_last.size[1]//8*8),Image.BICUBIC)
        #img_next = img_next.resize((img_next.size[0]//8*8, img_next.size[1]//8*8),Image.BICUBIC)
        img = cv2.resize(img,(1920, 1080),interpolation = cv2.INTER_CUBIC)
        gt_cnt = []
        for i in range(imgnum):
            gt_cnt += [np.sum(den[:,:,i])/100]

        return img.transpose((2,0,1)), np.array(gt_cnt)
'''
def cal_keypoints(keypoints, size):
    j, i, w, h = size
    nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)
    points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
    points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
    bbox = np.concatenate((points_left_up, points_right_down), axis=1)
    inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
    origin_area = nearest_dis * nearest_dis
    ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
    mask = (ratio >= 0.3)

    target = ratio[mask]
    keypoints = keypoints[mask]
    keypoints = keypoints[:, :2] - [j, i]  # change coodinate
    return keypoints, target

def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area

def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) #* 255.0，将数据转换回h,w,c
    image_numpy = image_numpy[:,:,0]#取第一通道？
    #image_numpy = np.transpose(image_numpy, (1, 2, 0))
    #image_gray = image_numpy[:,:,0] * 0.334 + image_numpy[:,:,1] * 0.333 + image_numpy[:,:,2] * 0.333
    #row,col = image_gray.shape
    '''
    plt.matshow(image_gray, cmap='hot')
    plt.colorbar()
    plt.xticks([])  
    plt.yticks([])
    #plt.show()
    plt.savefig("examples.jpg") 
    '''
    imsave(filename,image_numpy)
    sumnum = image_numpy.sum()/100
    #print ("Image:{}, count_pred:{} ".format(filename,int(sumnum), ))
    return sumnum
