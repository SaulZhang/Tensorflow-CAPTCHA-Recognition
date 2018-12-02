import matplotlib.pyplot as plt 
from PIL import Image,ImageFilter,ImageEnhance 
import numpy as np  
import random
from pylab import *
import os
from random import randint, choice

def segmentation_w(img):
    (w, h) = img.size 
    a = [0 for z in range(0, w)] 

    pix = img.load()

    for j in range(0, w):
        for i in range(0, h):
            if 0 <= pix[j, i] and pix[j, i] <= 140:
                a[j] += 1
    return a

def segmentation_h(img):
    (w, h) = img.size 
    a = [0 for z in range(0, h)] 

    pix = img.load()

    for j in range(0, w): 
        for i in range(0, h):
            if 0 <= pix[j, i] and pix[j, i] <= 140:
                a[i] += 1
    return a


def binarizing(img, threshold = 140):
    pixeldata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pixeldata[x, y] <= threshold:
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
    img = img.convert('L')
    binarizing(img)
    remove_hot_point(img)
    img = img.filter(ImageFilter.MedianFilter(size = 1))
    img = img.filter(ImageFilter.ModeFilter(size = 1))
    img = img.filter(ImageFilter.MedianFilter(size = 1))
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

def main(): 
    dirs = os.listdir('./data/train/')

    for filename in dirs:
        f = './data/train/'+ filename
        print(f)
        if os.path.isfile(f) and f.endswith('.jpg'):
            img = Image.open(f)
            img = FinalProcess(img)
            path = "./process_data/train/" + filename
            # print(path)
            img.save(path)

if __name__ == '__main__':
	main()