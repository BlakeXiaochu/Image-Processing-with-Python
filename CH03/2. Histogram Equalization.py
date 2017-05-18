import cv2, os
import numpy as np

#切换工作目录
os.chdir(r'D:\application\Coding\Image Processing\CH03\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03')

if os.getcwd() == r'D:\application\Coding\Image Processing\CH03\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03':
	img = cv2.imread('Fig0320(2)(2nd_from_top).tif')

	#统计直方图
	pr = np.zeros(256, dtype = 'float64')									#计算输入图像各灰度值概率
	for pixel in img.flat:
		pr[pixel] += 1
	pr /= img.size;

	Tr = pr.cumsum()														#计算变换函数
	Tr *= 255
	Tr = Tr.round()

	new_img = img.copy()													#均衡后的图像
	for pixel in np.nditer(new_img, op_flags = ['readwrite']):
		pixel[...] = Tr[pixel]


	cv2.imshow('New Img', new_img)											#显示直方图均衡后的图片
	cv2.waitKey(0)
	cv2.destroyAllWindows()