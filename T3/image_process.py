import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance 
import numpy as np  
import random
from pylab import *
import os
from random import randint, choice

def binarizing(img, threshold = 170):
    pixeldata = img.load()
    w,h = img.size
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
    img = ImageEnhance.Contrast(img).enhance(3.0)
    img = img.convert('L')
    binarizing(img)
    remove_hot_point(img)
    img = img.filter(ImageFilter.MedianFilter(size = 5))
    return img


def main(): 
    dirs = os.listdir('./data/train/')

    for filename in dirs:
        f = './data/train/'+ filename
        # print(f)
        if os.path.isfile(f) and f.endswith('.jpg'):
            img = Image.open(f)
            img = FinalProcess(img)
            path = "./process_data/train/" + filename
            # print(path)
            img.save(path)

if __name__ == '__main__':
	main()