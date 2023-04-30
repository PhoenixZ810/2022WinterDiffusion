'''生成高斯模糊之后的密度图以及h5py文件并进行对应'''
import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import json
import cv2
import pdb
from matplotlib import cm as CM
from decimal import Decimal

# from image import *

# set the root to the path of FDST dataset you download
# root = '/mnt/HDD2/zxy/2023winterdiffusion/data_dir/FDST/'
root = '/home/zxy/2023WinterDiffusion/data_dir/FDST/'
# now generate the FDST's ground truth
# train_folder = os.path.join(root,'try_traindata3N')
# test_folder = os.path.join(root,'try_testdata3N')
train_folder = os.path.join(root, 'train_data')
test_folder = os.path.join(root, 'test_data_small')
path_sets = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if
             os.path.isdir(os.path.join(train_folder, f))] + [os.path.join(test_folder, f) for f in
                                                              os.listdir(test_folder) if
                                                              os.path.isdir(os.path.join(test_folder, f))]

img_paths = []
# print(path_sets)
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)  # 将jpg路径名称添加到列表中
# print (img_paths)
img_row = 0

for img_path in img_paths:
    # print (img_path)
    # img_path = "/home/zxy/2023WinterDiffusion/data_dir/FDST/train_data/3/115.jpg"
    gt_path = img_path.replace('.jpg', '.json')
    # gt_path = "/home/zxy/2023WinterDiffusion/data_dir/FDST/test_data1/10/041.json"
    with open(gt_path, 'r') as f:  # 打开json文件
        gt = json.load(f)
    # print (gt)
    # break
    anno_list = list(gt.values())[0]['regions']
    # print (anno_list)
    # break
    # img= plt.imread(img_path)#读取图片
    # img_path = "/home/zxy/Bayes-loss++/datasets/UCF-QNRF_ECCV18/Train/img_0001.jpg"
    img = cv2.imread(img_path)[:, :, ::-1]
    im_h, im_w = img.shape[0], img.shape[1]
    # plt.imshow(img)
    # plt.show()
    # h_res, w_res = int(im_h), int(im_w)
    # img_res = img
    # h_res, w_res = int(im_h/2), int(im_w/2)
    # img_res = cv2.resize(img, (w_res, h_res), interpolation=cv2.INTER_AREA)
    # plt.imshow(img_res)
    # plt.show()
    h_res, w_res = int(im_h / 4), int(im_w / 4)
    if w_res % 32 != 0:
        w_res = int(Decimal(float(w_res) / 32).quantize(Decimal("0."), rounding="ROUND_HALF_UP") * 32)
    if h_res % 32 != 0:
        h_res = (int(h_res / 32)-1) * 32
    # pdb.set_trace()
    h_res, w_res = 160, 256
    img_res = cv2.resize(img, (w_res, h_res), interpolation=cv2.INTER_AREA)

    # plt.show()
    # h_res, w_res = int(im_h / 8), int(im_w / 8)
    # img_res = cv2.resize(img, (w_res, h_res), interpolation=cv2.INTER_AREA)
    # plt.imshow(img_res)
    # plt.show()

    # pdb.set_trace()
    # if im_w % 32 != 0:
    #     im_w = int(Decimal(float(im_w) / 32).quantize(Decimal("0."), rounding="ROUND_HALF_UP") * 32)
    #     resize = 1
    #
    # if im_h % 32 != 0:
    #     im_h = int(Decimal(float(im_h) / 32).quantize(Decimal("0."), rounding="ROUND_HALF_UP") * 32)
    #     resize = 1
    # if resize:
    #     img = cv2.resize(img, (im_w, im_h), interpolation=cv2.INTER_AREA)
    # plt.imshow(img)
    # plt.show()

    # if w ==1920 :
    #     img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    # cv2.imwrite(img_path, img)

    # h_resize = img.shape[0]
    # w_resize = img.shape[1]

    '''标点后插值resize'''
    # k = np.zeros((1080,1920))
    # for i in range(0,len(anno_list)):
    #     # y_anno = min(int(anno_list[i]['shape_attributes']['y']/rate_h),135)
    #     # x_anno = min(int(anno_list[i]['shape_attributes']['x']/rate_w),240)
    #     y_anno = int(anno_list[i]['shape_attributes']['y'])
    #     x_anno = int(anno_list[i]['shape_attributes']['x'])
    #     k[y_anno,x_anno]=1
    # k1 = cv2.resize(k, (k.shape[1] // 8, k.shape[0] // 8), interpolation=cv2.INTER_AREA) * 64
    # k2 = cv2.resize(k, (k.shape[1] // 8, k.shape[0] // 8), interpolation=cv2.INTER_AREA)
    # k3 = cv2.resize(k, (256, 160), interpolation=cv2.INTER_AREA) * (k.shape[1]/256) * (k.shape[0]/160)
    # pdb.set_trace()
    '''直接在标点的时候插值定位'''
    # h_den = int(h_resize/8)
    # w_den = int(w_resize/8)
    # # print(h,w)
    # k = np.zeros((h_den, w_den))
    # rate_h = int(1080/h_den)
    # rate_w = int(1920/w_den)
    # # rate_h = 8
    # # rate_w = 8
    # for i in range(0, len(anno_list)):
    #     y_anno = min(int(anno_list[i]['shape_attributes']['y'] / rate_h), 135)
    #     x_anno = min(int(anno_list[i]['shape_attributes']['x'] / rate_w), 240)
    #     k[y_anno, x_anno] = 1
    '''缩小倍数非整数关系在标点时差值定位'''
    h_den, w_den = h_res, w_res
    # print(h,w)
    k = np.zeros((h_den, w_den))
    GT = np.zeros((im_h, im_w))
    # k = np.zeros((1080, 1920))
    rate_h = im_h / h_den
    rate_w = im_w / w_den
    # rate_h = int(1080 / h_den)
    # rate_w = int(1920 / w_den)
    # rate_h = 8
    # rate_w = 8
    # '''原scale GT'''
    # for i in range(0, len(anno_list)):
    #     y_anno = int(anno_list[i]['shape_attributes']['y'])
    #     x_anno = int(anno_list[i]['shape_attributes']['x'])
    #     # y_anno = min(int(anno_list[i]['shape_attributes']['y'] / rate_h), 159)
    #     # x_anno = min(int(anno_list[i]['shape_attributes']['x'] / rate_w), 255)
    #     # y_anno = anno_list[i]['shape_attributes']['y']
    #     # x_anno = anno_list[i]['shape_attributes']['x']
    #     GT[y_anno, x_anno] = 1
    #     print(x_anno, y_anno)
    '''resize scale k'''
    for i in range(0, len(anno_list)):
        y_anno = min(int(float(anno_list[i]['shape_attributes']['y']) / rate_h)+3, h_den-1)
        x_anno = min(int(float(anno_list[i]['shape_attributes']['x']) / rate_w)+3, w_den-1)
        # y_anno = min(int(anno_list[i]['shape_attributes']['y'] / rate_h), 159)
        # x_anno = min(int(anno_list[i]['shape_attributes']['x'] / rate_w), 255)
        # y_anno = anno_list[i]['shape_attributes']['y']
        # x_anno = anno_list[i]['shape_attributes']['x']
        k[y_anno, x_anno] = 1
    # pdb.set_trace()
    k = gaussian_filter(k, 2)*5
    # '''GT与k对比'''
    # k, GT = gaussian_filter(k, 5) * 5, gaussian_filter(GT, 3) * 5
    # k, GT = np.expand_dims(k, axis=2), np.expand_dims(GT, axis=2)
    # plt.imshow(k)
    # plt.show()
    # plt.imshow(GT)
    # plt.show()
    # pdb.set_trace()

    with h5py.File(img_path.replace('.jpg', '.h5'), 'w') as hf:  # 创建以图片名+resize.h5为名的h5py文件并赋值density为密度图
        hf['image'] = img_res
        hf['density'] = k
        hf.close()
    img_row += 1
    if img_row % 100 == 0 or img_row == len(img_paths):
        print('process = {:.1%}'.format(img_row / len(img_paths)))
