from math import exp
import numpy as np
import cv2

#生成卷积核
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


if __name__ == '__main__':
	print()