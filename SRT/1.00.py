import cv2
import time
import numpy as np

#计算程序执行时间的装饰器
def time_test(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        location = fn(*args, **kwargs)
        print ("%s() cost %s second" % (fn.__name__, time.clock() - start))
        return location
    return _wrapper


@time_test
def img_choose(img_name, threshold = [80, 110], border = [25, 100]):

	deep_img = cv2.imread(img_name, 0)
	region = (threshold[0] < deep_img) & (deep_img < threshold[1])
	#region = np.zeros(deep_img.shape, dtype = 'bool')
	#region[threshold[0] < deep_img & deep_img < threshold[1]] = True

	shape = deep_img.shape
	height = np.arange(0, shape[0], 1)
	width = np.arange(0, shape[1], 1)

	location = [int(shape[0]/2), int(shape[1]/2), int(shape[0]/2), int(shape[1]/2)]			#top-left, bottom_right

	for h in height[border[0]: shape[0] - border[0]]:										#排除部分边缘
		for w in width[border[1]: shape[1] - border[1]]:
			if(region[h, w]):
				if(h < location[0] and region[h + 1 : h + 5, w].all()): location[0] = h
				if(w < location[1] and region[h, w + 1 : w + 5].all()): location[1] = w
				if(h > location[2] and region[h - 5 : h, w].all()): location[2] = h
				if(w > location[3] and region[h, w - 5 : w].all()): location[3] = w

	return location																			#return top-left', bottom_right' coordinate


if __name__ == '__main__':
	location = img_choose('da1.jpg', [50, 110])
	print(location)