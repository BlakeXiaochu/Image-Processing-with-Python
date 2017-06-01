import numpy as np
import cv2
import os
import time

#计算程序执行时间的装饰器
def time_test(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        result = fn(*args, **kwargs)
        print ("%s() cost %s second" % (fn.__name__, time.clock() - start))
        return result
    return _wrapper


@time_test
def Laplace(img, mode):
	if mode == 'four':
		kernal = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0], dtype = 'int32')
	elif mode == 'eight':
		kernal = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype = 'int32')
	else:
		print(self.__name__ , 'Mode Error')
		return None
	#kernal = -kernal
	#kernel.shape = 3, 3

	#扩展边缘
	img_copy = np.zeros([img.shape[0] + 2, img.shape[1] + 2], dtype = 'int32')
	img_copy[1 : img.shape[0] + 1, 1 : img.shape[1] + 1] = img

	new_img = np.zeros(img.shape, dtype = 'int32')
	it = np.nditer(new_img, op_flags = ['readwrite'], flags = ['multi_index'])
	while not it.finished:
		x = it.multi_index[0]
		y = it.multi_index[1]
		neighbor = img_copy[x : x + 3, y : y + 3]
		#互相关操作
		it[0] = np.correlate(neighbor.ravel(), kernal) 										
		it.iternext()

	#阈值处理
	new_img += img
	new_img[new_img < 0] = 0
	new_img[new_img > 255] = 255

	return new_img.astype('uint8')


@time_test
def  Unsharp_Masking(img, degree = 1.0, kernal_size = [3, 3]):
	mean_filter = lambda kernal_size: np.ones(kernal_size, dtype = 'float') / (kernal_size[0]*kernal_size[1])			
	kernal = mean_filter(kernal_size)
	kernal = kernal.ravel()

	shape = img.shape
	new_shape = [shape[0] + (kernal_size[0] - 1), shape[1] + (kernal_size[1] - 1)]
	#边界外部分区域扩展为0值区域
	img_copy = np.zeros(new_shape, dtype = 'float')
	img_copy[(kernal_size[0] - 1)//2 : new_shape[0] - (kernal_size[0] - 1)//2, (kernal_size[1] - 1)//2 : new_shape[1] - (kernal_size[1] - 1)//2] = img

	mask = img.astype('float')
	#旋转卷积核
	#kernal = kernal[::-1, ::-1]
	it = np.nditer(mask, op_flags = ['readwrite'], flags = ['multi_index'])
	while not it.finished:
		x = it.multi_index[0]
		y = it.multi_index[1]
		neighbor = img_copy[x : x + kernal_size[0], y : y + kernal_size[1]]
		it[0] = np.correlate(neighbor.ravel(), kernal)													#互相关操作
		it.iternext()

	mask = img - mask
	#return np.fabs(degree*mask).astype('uint8')
	new_img = (img + degree*mask).clip(0, 255).astype('uint8')
	return new_img


@time_test
def Gradiant():
	pass



if __name__ == '__main__':
	#切换工作目录
	os.chdir(r'D:\application\Coding\Image Processing\CH03\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03')
	img = cv2.imread('Fig0338(a)(blurry_moon).tif', 0)

	#new_img = Laplace(img, mode = 'eight')
	new_img = Unsharp_Masking(img, 3)

	#显示锐化后的图片
	cv2.imshow('New Img', new_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
