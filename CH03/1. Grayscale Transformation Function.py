import cv2, os
import numpy as np


#切换工作目录
os.chdir(r'D:\application\Coding\Image Processing\CH03\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03')

if os.getcwd() == r'D:\application\Coding\Image Processing\CH03\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03':
	print('Transformation Function:\n1. inverse\n2. logarithmic\n3. idempotent\n')
	trans_type = input('Choose: ')
	#读取图片
	img = cv2.imread('Fig0304(a)(breast_digital_Xray).tif', 0)
	if trans_type == '1':													#反转变换
		f = lambda img: 255 - img\
		new_img = f(img)

	elif trans_type == '2':													#对数变换
		c = 255/math.log(2)/8
		new_img = img.astype('int32')
		new_img += 1
		new_img = np.log(new_img)
		new_img *= c
		new_img = new_img.astype('uint8')


	elif trans_type == '3':													#幂等变换
		degree = float(input('\nDegree: '))									#变换的阶数
		f = lambda img: img**degree/(255**(degree - 1))
		new_img = f(img.astype('int32')).astype('uint8')

	else:																	#输入无效
		print('Invalid Input')
		exit()
		new_img = img.copy().fill(0)	

	cv2.imshow('New Img', new_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()