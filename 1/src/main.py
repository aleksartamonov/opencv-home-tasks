import random

__author__ = 'aleksart'

import cv2
import reader
import numpy as np
import random

class Descriptor:
    def __init__(self, kp, des, filename):
        self.kp = kp
        self.des = des
        self.filename = filename


files = reader.load_data('../data')
descriptors = []
k = 0
for filename in files:
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray, None)
    for index in xrange(len(kp)):
        descriptors.append(Descriptor(kp[index], des[index], filename))
    k += 1
    print k

kmeans_input_des = np.vstack([x.des for x in descriptors])

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
K = 100
ret, label, center = cv2.kmeans(kmeans_input_des, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
for clas in xrange(5):
    label_num = random.choice(range(K))
    temp_instances = []
    for index in xrange(len(descriptors)):
        if label.ravel()[index] == label_num:
            temp_instances.append(descriptors[index])
    size = 50
    class_images = []
    inst_num = 0
    while inst_num < 10:
        inst = random.choice(temp_instances)
        img = cv2.imread(inst.filename)
        img_with_key = cv2.drawKeypoints(img, [inst.kp])
    # key_image = np.zeros((50,50,3), np.uint8)
        key_image = img_with_key[inst.kp.pt[0] - size/2:inst.kp.pt[0] + size/2, inst.kp.pt[1] - size/2:inst.kp.pt[1] + size/2]
        if key_image.shape == (50,50,3) :
            class_images.append(key_image)
            inst_num += 1
        print label_num, inst_num
    class_image = np.concatenate(class_images)
    cv2.imwrite('../result/'+str(clas)+'.jpg', class_image)

