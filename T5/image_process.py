import matplotlib.pyplot as plt
from PIL import Image,ImageFilter,ImageEnhance
import numpy as np
import random
from pylab import *
import os
from random import randint, choice

old_dir = "./data/train/"
save_dir = "./process_data/train/"

# SIZE_VALIDATION_SET = 2000

def segmentation_w(img):
    (w,h) = img.size
    a = [0 for z in range(0, w)] 

    pix = img.load()

    for j in range(0, w): 
        for i in range(0, h):
            if 0 <= pix[j, i] and pix[j, i] <= 143:
                a[j] += 1

    return a

def segmentation_h(img):
    (w, h) = img.size
    a = [0 for z in range(0, h)] 

    pix = img.load()

    for j in range(0, w):
        for i in range(0, h):
            if 0 <= pix[j, i] and pix[j, i] <= 143:
                a[i] += 1

    return a


def binarizing(img, Minthreshold = 0, Maxthreshold = 143):
	pixeldata = img.load()
	w,h = img.size
	for y in range(h):
		for x in range(w):
			if pixeldata[x, y] <= Maxthreshold and pixeldata[x, y] >= Minthreshold:
				pixeldata[x, y] = 0
			else:
				pixeldata[x, y] = 255 


def remove_hot_point(img):
	pixeldata = img.load()
	w,h = img.size
	for y in range(0, h):
		for x in range(0, w):
			count = 0
			if y - 1 >= 0 and pixeldata[x, y - 1] > 240:
				count += 1
			if y + 1 < h and pixeldata[x, y + 1] > 240:
				count += 1
			if x - 1 >= 0 and pixeldata[x - 1, y] > 240:
				count += 1
			if x + 1 < w and pixeldata[x + 1, y] > 240:
				count += 1	
			if count > 2:
				pixeldata[x, y] = 255


def FinalProcess(img):
	img = ImageEnhance.Contrast(img).enhance(2.0)
	img = img.convert('L')
	binarizing(img)
	remove_hot_point(img)
	img = img.filter(ImageFilter.MedianFilter(size=1))
	remove_hot_point(img)
	img = img.filter(ImageFilter.ModeFilter(size = 1))
	img = img.filter(ImageFilter.MedianFilter(size = 1))
	img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
	W = segmentation_w(img)
	H = segmentation_h(img)
	pixeldata = img.load()
	w,h = img.size
	for i in range(w):
		if (W[i] <= 2):
			for j in range(h):
				pixeldata[i, j] = 255
	for i in range(h):
		if (H[i] <= 2):
			for j in range(w):
				pixeldata[j, i] = 255
	for i in range(w):
		for j in range(h):
			if (i - 5 < 0 or i + 5 >= w or j - 5 <= 0 or j + 5 >= h):
				pixeldata[i, j] = 255
	remove_hot_point(img)
	binarizing(img)
	return img

def cropT5(img, count, label, a):
	path = save_dir
	dir_ = path + a + str(count)
	if not os.path.exists(dir_):
		os.makedirs(dir_)
	img_crop = img.crop((0, 0, 45, 45))
	img_crop = FinalProcess(img_crop)
	img_crop.save(dir_ + '/' + '9_0-' + label[0] + ".jpg")

	img_crop = img.crop((34, 0, 79, 45))
	img_crop = FinalProcess(img_crop)
	img_crop.save(dir_ + '/' + '9_1-' + label[1] + ".jpg")
	
	img_crop = img.crop((71, 0, 116, 45))
	img_crop = FinalProcess(img_crop)
	img_crop.save(dir_ + '/' + '9_2-' + label[2] + ".jpg")
	
	img_crop = img.crop((105, 0, 150, 45))
	img_crop = FinalProcess(img_crop)
	img_crop.save(dir_ + '/' + '9_3-' + label[3] + ".jpg")


def main():
	count = 0
	text = open("./process_data/mappings.txt")
	# for i in range(SIZE_VALIDATION_SET):
	# 	line = text.readline()
	while True:
		if count < 10:
			a = "000"
		elif count < 100:
			a = "00"
		elif count < 1000:
			a = "0"
		else: 
			a = ""
		line = text.readline()
		label = line.split(',')[-1][0:4]
		print(count, label)
		path = old_dir + a + str(count) + '/' + a + str(count) + ".jpg"
		img = Image.open(path)
		img = FinalProcess(img)
		cropT5(img, count, label, a)
		for i in range(9):
			path1 = old_dir + a + str(count) + '/' + str(i) + ".jpg"
			im = Image.open(path1)
			im = FinalProcess(im)
			path2 = save_dir + a + str(count) + "/" +  str(i) + ".jpg"
			im.save(path2)

		count += 1
		if count == 9900:
			break
	text.close()

if __name__=='__main__':
	main()