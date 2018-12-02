import random
import os
from PIL import Image,ImageFilter

SAVE_PATH = './process_data/train/'

SIZE_VALIDATION_SET = 9900

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
	img = img.convert('L')
	binarizing(img)
	remove_hot_point(img)
	img = img.filter(ImageFilter.EDGE_ENHANCE)
	img = img.filter(ImageFilter.ModeFilter(size = 5))
	img = img.filter(ImageFilter.MedianFilter(size = 7))
	return img

def segmentation(img):
    w,h = img.size 
    a = [0 for z in range(0, w)] 
    pix = img.load()
    for j in range(0, w):
        for i in range(0, h):
            if 0 <= pix[j, i] and pix[j, i] <= 170:
                a[j] += 1     
    return a

def crop(img, a, text):
    w,h = img.size
    step = 0
    char_len = 36
    char_hei = 48
    startx, starty = 0, 19
    count = 0
    Maxlen = len(text)
    w,h = img.size
    while True:
        if step + 36 >= w:
            break
        if a[step] >= 4 and a[step + 10] >= 4 and a[step + 18] >= 4:
            img_crop = img.crop((startx, starty, startx + char_len, starty + char_hei))
            char = text[count]
            if char == '*':
                char = '×'
            img_crop.save(SAVE_PATH + char + '/' + str(random.randint(0,1e5)) + '-' + str(random.randint(0,1e5)) + '.jpg')
            step += 36
            if count < Maxlen - 1: 
                count += 1
        step += 1
        startx = step

def crop_for_test(img, a):
    w,h = img.size
    step = 0
    char_len = 36
    char_hei = 48
    startx, starty = 0, 19
    images = []
    while True:
        if step + 36 >= w:
            break
        if a[step] >= 4 and a[step + 10] >= 4 and a[step + 18] >= 4 and a[step + 2] >= 4 and a[step + 4] >= 4 and a[step + 6] >= 4:
            img_crop = img.crop((startx, starty, startx + char_len, starty + char_hei))
            images.append(img_crop)
            step += 36
        step += 1
        startx = step
    return images 

def main():
    file = open('./process_data/mappings.txt')
    # for i in range(SIZE_VALIDATION_SET):
    #     text = file.readline()
    dirs = os.listdir('./data/train/')
    count = 0;

    for filename in dirs:
      f = './data/train/' + filename
      if os.path.isfile(f) and f.endswith('.jpg'):
          img = Image.open(f)
          img = FinalProcess(img)
          path = './process_data/train/' + filename
          # print(path)
          # img.save(path)
          text = file.readline().split(',')[-1].split('=')[0]
          crop(img,segmentation(img),text)
          print(filename)
          count += 1
          if count == SIZE_VALIDATION_SET:
            break
    file.close()

if __name__ == '__main__':
	main()