import numpy as np
import cv2


def sharpen(img):
	kernel = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0], dtype = 'float')
	kernel.shape = 3, 3
	new_img = np.zeros(img.shape, dtype = 'float')


if __name__ == '__main__':
	#切换工作目录
	os.chdir(r'D:\application\Coding\Image Processing\CH03\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03')
	img = cv2.imread('Fig0333(a)(test_pattern_blurring_orig).tif', 0)
