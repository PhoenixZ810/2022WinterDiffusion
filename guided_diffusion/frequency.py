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

'''生成圆形mask'''
def create_circular_mask(h, w, center=None, radius=None, in_outside = 'in'):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((Y - center[0])**2 + (X-center[1])**2)
    if in_outside == 'in':
        mask = dist_from_center <= radius
    else:
        mask = dist_from_center >= radius
    return mask

# print (img_path)
img_path = "/home/zxy/2023WinterDiffusion/data_dir/FDST/test_data1_small/10/092.jpg"
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
img = cv2.imread(img_path)
im_h, im_w = img.shape[0], img.shape[1]

'''对RGB图像进行fft变换'''
# plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)
img_c1 = cv2.imread(img_path, 0)
img_c2 = np.fft.fft2(img_c1)
img_c3 = np.fft.fftshift(img_c2)

rows, cols = img_c1.shape
crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
'''高通'''
# mask = np.ones((rows, cols), np.uint8)
# mask[crow - 20:crow + 20, ccol - 20:ccol + 20] = 0
# f_high = img_c3 * mask
mask = create_circular_mask(rows, cols, (crow, ccol), 50, 'in')
f_high = img_c3.copy()
f_high[mask] = 0

'''低通'''
mask = np.zeros((rows, cols), np.uint8)
mask[crow - 100:crow + 100, ccol - 100:ccol + 100] = 1
f_low = img_c3 * mask

img_c4 = np.fft.ifftshift(img_c3)
img_c5 = np.fft.ifft2(img_c4)
img_c6 = np.fft.ifftshift(f_high)
img_c7 = np.fft.ifft2(img_c6)
img_c8 = np.fft.ifftshift(f_low)
img_c9 = np.fft.ifft2(img_c8)
'''边缘检测'''
canny1 = cv2.Canny(img_c1, 50, 150)
canny2 = cv2.Canny(img_c9.astype(np.uint8), 10, 100)
# plt.subplot(331), plt.imshow(img_c1, "gray"), plt.title("Original Image")
# plt.subplot(332), plt.imshow(np.log(1 + np.abs(img_c2)), "gray"), plt.title("Spectrum")
# plt.subplot(333), plt.imshow(np.log(1 + np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")
# plt.subplot(334), plt.imshow(np.log(1 + np.abs(img_c4)), "gray"), plt.title("Decentralized")
# plt.subplot(335), plt.imshow(np.abs(img_c5), "gray"), plt.title("Inverse Image")
# plt.subplot(336), plt.imshow(np.log(1 + np.abs(img_c4)), "gray"), plt.title("Decentralized")
# plt.subplot(337), plt.imshow(np.abs(img_c5), "gray"), plt.title("Highpass Image")
plt.imshow(img_c1, "gray"), plt.title("Original Image")
plt.show()
plt.imshow(np.log(1 + np.abs(img_c2)), "gray"), plt.title("Spectrum")
# plt.show()
plt.imshow(np.log(1 + np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")
plt.show()
plt.imshow(np.log(1 + np.abs(img_c4)), "gray"), plt.title("Decentralized")
# plt.show()
plt.imshow(np.abs(img_c5), "gray"), plt.title("Inverse Image")
# plt.show()
plt.imshow(np.log(1 + np.abs(img_c6)), "gray"), plt.title("Decentralized")
# plt.show()
plt.imshow(np.abs(img_c7), "gray"), plt.title("Highpass Image")
plt.show()
plt.imshow(canny1, "gray"), plt.title("Edge Original Image")
plt.show()
plt.imshow(canny2, "gray"), plt.title("Edge Highpass Image")
plt.show()
plt.imshow(np.log(1 + np.abs(img_c8)), "gray"), plt.title("Decentralized")
# plt.show()
plt.imshow(np.abs(img_c9), "gray"), plt.title("Lowpass Image")
plt.show()

'''缩小倍数非整数关系在标点时差值定位'''
h_den = 160
w_den = 256
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
'''原scale GT'''
for i in range(0, len(anno_list)):
    y_anno = int(anno_list[i]['shape_attributes']['y'])
    x_anno = int(anno_list[i]['shape_attributes']['x'])
    # y_anno = min(int(anno_list[i]['shape_attributes']['y'] / rate_h), 159)
    # x_anno = min(int(anno_list[i]['shape_attributes']['x'] / rate_w), 255)
    # y_anno = anno_list[i]['shape_attributes']['y']
    # x_anno = anno_list[i]['shape_attributes']['x']
    GT[y_anno, x_anno] = 1
'''resize scale k'''
for i in range(0, len(anno_list)):
    y_anno = min(int(float(anno_list[i]['shape_attributes']['y']) / rate_h), 159)
    x_anno = min(int(float(anno_list[i]['shape_attributes']['x']) / rate_w), 255)
    # y_anno = min(int(anno_list[i]['shape_attributes']['y'] / rate_h), 159)
    # x_anno = min(int(anno_list[i]['shape_attributes']['x'] / rate_w), 255)
    # y_anno = anno_list[i]['shape_attributes']['y']
    # x_anno = anno_list[i]['shape_attributes']['x']
    k[y_anno, x_anno] = 1
# pdb.set_trace()
# k = gaussian_filter(k,3)
k, GT = gaussian_filter(k, 3) * 5, gaussian_filter(GT, 10) * 5
k, GT = np.expand_dims(k, axis=2), np.expand_dims(GT, axis=2)

'''k进行频域变换'''
img_c1 = k
img_c2 = np.fft.fft2(img_c1)
img_c3 = np.fft.fftshift(img_c2)
img_c4 = np.fft.ifftshift(img_c3)
img_c5 = np.fft.ifft2(img_c4)
plt.subplot(231), plt.imshow(img_c1, "gray"), plt.title("Original Image")
plt.subplot(232), plt.imshow(np.log(1 + np.abs(img_c2)), "gray"), plt.title("Spectrum")
plt.subplot(233), plt.imshow(np.log(1 + np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")
plt.subplot(234), plt.imshow(np.log(1 + np.abs(img_c4)), "gray"), plt.title("Decentralized")
plt.subplot(235), plt.imshow(np.abs(img_c5), "gray"), plt.title("Processed Image")
plt.show()
plt.imshow(k, cmap=CM.jet)
plt.show()
plt.imshow(np.log(1 + np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")
plt.show()
'''GT进行频域变换'''
img_c1 = GT
img_c2 = np.fft.fft2(img_c1)
img_c3 = np.fft.fftshift(img_c2)
img_c4 = np.fft.ifftshift(img_c3)
img_c5 = np.fft.ifft2(img_c4)
plt.suptitle('GT')
plt.subplot(231), plt.imshow(img_c1, "gray"), plt.title("Original Image")
plt.subplot(232), plt.imshow(np.log(1 + np.abs(img_c2)), "gray"), plt.title("Spectrum")
plt.subplot(233), plt.imshow(np.log(1 + np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")
plt.subplot(234), plt.imshow(np.log(1 + np.abs(img_c4)), "gray"), plt.title("Decentralized")
plt.subplot(235), plt.imshow(np.abs(img_c5), "gray"), plt.title("Processed Image")
plt.tight_layout()
plt.show()
plt.imshow(GT, cmap=CM.jet)
plt.show()
plt.imshow(np.log(1 + np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")
plt.show()