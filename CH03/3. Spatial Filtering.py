from math import exp
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


#生成卷积核
@time_test
def filter_kernal(kernal_type, kernal_size = (3, 3)):
	#均值滤波器
	mean_filter = lambda kernal_size: np.ones(kernal_size, dtype = 'float') / (kernal_size[0]*kernal_size[1])			
	#加权平均滤波器（高斯）
	def gauss_filter(kernal_size):
		#高斯函数
		def gauss_func(x, y):
			x_c = (kernal_size[0] - 1)/2 - x
			y_c = (kernal_size[1] - 1)/2 - y
			return (0.5)**(x_c**2 + y_c**2)

		kernal = np.fromfunction(gauss_func, kernal_size, dtype = 'float')
		return kernal / kernal.sum()															
	#滤波器类型
	Filter_Type = {
		'mean' : mean_filter,
		'gauss' : gauss_filter
	}
	#生成卷积核
	try:
		kernal = Filter_Type[kernal_type](kernal_size)
	except KeyError as e:
		print(e, ": There's no filter type of " + kernal_type)
		return None
	else:
		return kernal


#卷积操作
#@time_test
def convolution(img, kernal, x, y, kernal_size = None):
	if(not kernal_size): kernal_size = kernal.shape

	#x, y = postion
	neighbor = img[x - (kernal_size[0] - 1)//2 : x + (kernal_size[0] - 1)//2 + 1, y - (kernal_size[1] - 1)//2 : y + (kernal_size[1] - 1)//2 + 1]

	#累计求和
	return (neighbor*kernal).sum()


#滤波器
@time_test
def filter(img, kernal_type, kernal_size):
	shape = img.shape
	new_shape = [shape[0] + (kernal_size[0] - 1), shape[1] + (kernal_size[1] - 1)]
	#边界外部分区域扩展为0值区域
	img_copy = np.zeros(new_shape, dtype = 'float')
	img_copy[(kernal_size[0] - 1)//2 : new_shape[0] - (kernal_size[0] - 1)//2, (kernal_size[1] - 1)//2 : new_shape[1] - (kernal_size[1] - 1)//2] = img

	new_img = np.zeros(shape, dtype = 'float')
	kernal = filter_kernal(kernal_type, kernal_size)
	#旋转卷积核
	kernal = kernal[::-1, ::-1]
	
	'''
	x_axis = np.arange((kernal_size[0] - 1)//2, new_shape[0] - (kernal_size[0] - 1)//2)
	y_axis = np.arange((kernal_size[1] - 1)//2, new_shape[1] - (kernal_size[1] - 1)//2)
	for x in x_axis:
		for y in y_axis:
			#new_img[x - (kernal_size[0] - 1)//2, y - (kernal_size[1] - 1)//2] = y - x
			new_img[x - (kernal_size[0] - 1)//2, y - (kernal_size[1] - 1)//2] = convolution(img_copy, kernal, x, y, kernal.shape)
	'''


	it = np.nditer(new_img, op_flags = ['readwrite'], flags = ['multi_index'])
	while not it.finished:
		#it[0] = convolution(img_copy, kernal, it.multi_index[0] + (kernal_size[0] - 1)//2, it.multi_index[1] + (kernal_size[1] - 1)//2, kernal.shape)
		x = it.multi_index[0]
		y = it.multi_index[1]
		neighbor = img_copy[x : x + kernal_size[0], y : y + kernal_size[1]]
		#it[0] = np.correlate(neighbor.flatten(), kernal.flatten())
		it[0] = np.correlate(neighbor.ravel(), kernal.ravel())												#互相关操作
		it.iternext()

	return new_img.round().astype(dtype = 'uint8')


if __name__ == '__main__':
	#切换工作目录
	os.chdir(r'D:\application\Coding\Image Processing\CH03\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03')
	img = cv2.imread('Fig0333(a)(test_pattern_blurring_orig).tif', 0)

	new_img = filter(img, 'mean', [11, 11])

	#显示滤波后的图片
	cv2.imshow('New Img', new_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()