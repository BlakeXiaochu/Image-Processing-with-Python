import cv2
import time
import numpy as np

#计算程序执行时间的装饰器
def time_test(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        location, max_pos = fn(*args, **kwargs)
        print ("%s() cost %s second" % (fn.__name__, time.clock() - start))
        return location, max_pos
    return _wrapper


@time_test
def img_choose(deep_img, threshold = 150, border = [25, 50]):

	region = deep_img > threshold
	#region = np.zeros(deep_img.shape, dtype = 'bool')
	#region[threshold[0] < deep_img & deep_img < threshold[1]] = True

	shape = deep_img.shape
	height = np.arange(0, shape[0], 1)
	width = np.arange(0, shape[1], 1)

	location = [int(shape[0]/2), int(shape[1]/2), int(shape[0]/2), int(shape[1]/2)]			#top-left, bottom_right
	max_value = threshold
	max_pos = [0, 0]

	for h in height[border[0]: shape[0] - border[0]]:										#排除部分边缘
		for w in width[border[1]: shape[1] - border[1]]:
			if(region[h, w]):
				if(deep_img[h, w] > max_value):
					if((deep_img[h - 4 : h + 4, w - 4 : w + 5] > 180).all()):
						max_value = deep_img[h, w]
						max_pos = [h, w]
				if(h < location[0] and region[h + 1 : h + 5, w].all()): location[0] = h
				if(w < location[1] and region[h, w + 1 : w + 5].all()): location[1] = w
				if(h > location[2] and region[h - 5 : h, w].all()): location[2] = h
				if(w > location[3] and region[h, w - 5 : w].all()): location[3] = w

	return location, max_pos																#return top-left', bottom_right' coordinate


if __name__ == '__main__':
	name_part = ['a', 'b', 'c', 'd', 'e']
	with open('Bounding Box.txt', mode = 'w') as output:
		for i in range(0, 5):
			for j in range(1, 6):
				deep_img_name = 'x' + 'd' + name_part[i] + str(j) + '.jpg'
				color_img_name = 'x' + 'c' + name_part[i] + str(j) + '.jpg'
				deep_img = cv2.imread(deep_img_name, 0)
				color_img = cv2.imread(color_img_name)
				location, max_pos = img_choose(deep_img, threshold = 150)

				h_mapping = lambda x: int((x - 47.10)/0.7708)
				w_mapping = lambda x: int((x - 22.35)/0.7022)

				#max_pos = deep_img.argmax()														#根据最近点，标定工件上的区域位置
				#h_max_pos = int(max_pos/640 - 1)
				#w_max_pos = int(max_pos - 640*(h_max_pos + 1) - 1)
				h_max_pos = max_pos[0]
				w_max_pos = max_pos[1]
				deep_img[:, w_max_pos] = 255
				deep_img[h_max_pos, :] = 255

				h_max_pos = h_mapping(h_max_pos)
				w_max_pos = w_mapping(w_max_pos)
				color_img[:, w_max_pos] = np.array([255, 255, 255])
				color_img[h_max_pos, :] = np.array([255, 255, 255]) 

				#print('Depth : ', location)
				cv2.imwrite('new_' + deep_img_name, deep_img[location[0]:location[2], location[1]:location[3]])

				location[0] = h_mapping(location[0])											#映射深度图像坐标至彩色图像
				location[2] = h_mapping(location[2])
				location[1] = w_mapping(location[1])
				location[3] = w_mapping(location[3])
				output.write('(' + str(location[0]) + ',' + str(location[2]) + ')' + ' ' + '(' + str(location[1]) + ',' + str(location[3]) + ')' + '\n')
				#print('Color : ', location, '\n')
				cv2.imwrite('new_' + color_img_name, color_img[location[0]:location[2], location[1]:location[3]])