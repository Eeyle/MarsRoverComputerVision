import os
from PIL import Image

images_path = '../data/test/'
for f in os.listdir(images_path):
	if f.endswith('.JPG'):
		img = Image.open(images_path + f).convert('RGB').convert('L') # this extra conversion handily removes alpha values
		img = img.resize((64, 64), Image.BILINEAR)
		img.save(images_path + f)

