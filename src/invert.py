import cv2
import os

data_dir = '../../data/fnt'
data_sub = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]

os.mkdir('../../data/new_fnt/')

for d in data_sub:
    imgs = [os.path.join(d, im) for im in os.listdir(d)]
    for i in imgs:
        im = cv2.imread(i)
        iminv = cv2.bitwise_not(im)
        fname = i.split('/')[-1].split('.')[0] + '.png'
        fclass = i.split('/')[-2]
        new_dir = os.path.join('../../data/new_fnt/', fclass)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        cv2.imwrite(os.path.join(new_dir, fname), iminv)