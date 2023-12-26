# -*- coding: utf-8 -*-

from PIL import Image
import os
import cv2

crop_w = 512   #裁剪图像宽度
crop_h = 512   #裁剪图像高度

imageDir = "E:\\dataset1\\AerialImageDataset\\test\\tig2jpg"  #./Original/Images/Labels     #原大图像数据
saveDir = "E:\\dataset1\\AerialImageDataset\\test\cut" + str(crop_w) + "x" + str(crop_h) + "/image/"    ##裁剪小图像数据
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

for name in os.listdir(imageDir):

    img = cv2.imread(imageDir + name)
    old_size= img.shape[0:2]   #原图尺寸
    # print(old_size[0],type(old_size[0]))
    ######
    h_num = int(old_size[0]/crop_h) + 1   #取整后加1
    w_num =int(old_size[1]/crop_w) + 1
    new_height = (h_num) * crop_h     #小图像尺寸整倍数的大图像
    new_width = (w_num) * crop_w
    print(new_height,new_width)
    # #
    pad_h = new_height - old_size[0]  # 计算自动需要填充的像素数目（图像的高这一维度上）
    pad_w = new_width - old_size[1]  # 计算需要填充的像素数目（图像的宽这一维度上）
    print(pad_w,pad_h)
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    print(top, bottom, left, right)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    #
    # kh = int(new_height/dish) - 1
    # kw = int(new_width/disw) - 1
    # print(kh,kw)
    for i in range(h_num):
        for j in range(w_num):
            # print(i,j)
            x = int(i * crop_h)  #高度上裁剪图像个数
            y = int(j * crop_w)
            print(x,y)
            img_crop = img_new[x : x + crop_h,y : y + crop_w]
            # print(z)
            saveName= name.split('.')[0] + '-' + str(i) +'-'+ str(j) +".png"  #小图像名称，内含小图像的顺序
            cv2.imwrite(saveDir+saveName,img_crop)
