import random
import os
from PIL import Image, ImageFilter, ImageEnhance

SAVE_PATH = "./process_data/train/"

SIZE_VALIDATION_SET = 9900

def binarizing(img, threshold = 95):
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
	for y in range(0,h):
		for x in range(0,w):
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
	img = ImageEnhance.Contrast(img).enhance(3.5)
	img = img.convert('L')
	binarizing(img)
	remove_hot_point(img)
	remove_hot_point(img)
	remove_hot_point(img)
	img = img.filter(ImageFilter.MedianFilter(size = 3))
	img = img.filter(ImageFilter.MinFilter(size = 1))
	img = img.filter(ImageFilter.MedianFilter(size = 1))
	remove_hot_point(img)
	img = img.filter(ImageFilter.SMOOTH)
	remove_hot_point(img)
	img = img.filter(ImageFilter.SMOOTH)
	binarizing(img)

	return img

def segmentation(img):
    w,h = img.size 
    a = [0 for z in range(0, w)] 
    pix = img.load()
    for i in range(0, w): 
        for j in range(0, h):  
            if pix[i, j] <= 170:  
                a[i] += 1            
    return a

def crop(img, a, text):
	w,h = img.size
	maxlen = len(text)	
	char_len = 34
	char_hei = 44
	startx,starty = 5, 10
	step = 0
	count = 0

	while True:	
		#if count == maxlen:break
		if count == 4:
			img_crop = img.crop((startx, starty, 200, starty + char_hei)).resize((64, 64), Image.ANTIALIAS)
			char = text[count]
			img_crop.save(SAVE_PATH + char + '/' + str(random.randint(0, 1e5)) + '-' + str(random.randint(0, 1e5)) + '.jpg')
			break
		if a[step] >= 2 and a[step+1] >= 2 and a[step + 2] >= 2 and a[step + 3] >= 5 and a[step + 4] >= 5 and a[step + 5] >= 4 and a[step + 6] >= 3:
			img_crop = img.crop((startx, starty, startx + char_len, starty + char_hei)).resize((64, 64), Image.ANTIALIAS)
			char = text[count]
			img_crop.save(SAVE_PATH + char + '/' + str(random.randint(0, 1e5)) + '-' + str(random.randint(0, 1e5)) + '.jpg')
			step += (char_len - 4)
			count += 1
		step += 1
		startx = step

def crop_for_test(img, a):
	w,h = img.size 
	char_len = 36
	char_hei = 44
	startx,starty=5,10
	images = [] 
	step = 0
	count = 0

	while True: 
		if count == 4:
			img_crop = img.crop((startx, starty, 200, starty + char_hei)).resize((64, 64), Image.ANTIALIAS)
			images.append(img_crop)
			break
		if a[step] >= 2 and a[step + 1] >= 2 and a[step + 2] >= 2 and a[step + 3] >= 4 and a[step + 4] >=4 and a[step + 5] >= 4 and a[step + 6] >= 2:
			img_crop = img.crop((startx, starty, startx + char_len, starty + char_hei)).resize((64, 64), Image.ANTIALIAS)
			images.append(img_crop)
			step += (char_len - 4)
			count += 1
		step += 1
		startx = step
	return images 

def main():	
	file = open('./process_data/mappings.txt')
	# for i in range(SIZE_VALIDATION_SET):
		# text = file.readline()
	dirs = os.listdir('./data/train/')

	for filename in dirs:
		f = './data/train/'+ filename
		# print(f)
		if os.path.isfile(f) and f.endswith('.jpg'):
			img = Image.open(f)
			img = FinalProcess(img)
			path = "./process_data/train/" + filename
			# print(path)
			
			# img.save(path)
			text = file.readline().split(',')[-1][0:5]
			crop(img, segmentation(img), text)
			print(filename)
	file.close()

if __name__ == '__main__':
	main()
 