__author__ = 'aleksart'


from reader import load_data
import random
import numpy as np
import cv2
from printer import print_to_arff

class SiftDescriptor:
    def __init__(self, kp, des, filename):
        self.kp = kp
        self.des = des
        self.filename = filename
        self.class_ = filename.split('/')[-2]


class BagDescriptor:
    beans = []

    def __init__(self, k, key):
        self.beans = [0 for x in xrange(k)]
        self.key = key
        self.class_ = key.split('/')[-2]

    def __str__(self):
        self.normalize_beans()
        out_str = ','.join([str(i) for i in self.norm_beans])
        out_str += "," + self.class_
        return out_str

    def normalize_beans(self):
        s = sum(self.beans)
        self.norm_beans = [i*1e0/s for i in self.beans]


if __name__ == "__main__":

    files = load_data('../../1/data')

    descriptors = []
    k = 0
    for filename in files:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(gray, None)
        for index in xrange(len(kp)):
            descriptors.append(SiftDescriptor(kp[index], des[index], filename))
        k += 1
        print k

    kmeans_input_des = np.vstack([x.des for x in descriptors])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    K = 100
    ret, label, center = cv2.kmeans(kmeans_input_des, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    bagDescriptors = dict()
    allClasses = set()
    for index in xrange(len(descriptors)):
        key = descriptors[index].filename
        temp_label = label[index]
        allClasses.add(descriptors[index].class_)
        if not key in bagDescriptors.keys():
            bagDescriptor = BagDescriptor(K, key)
            bagDescriptor.beans[temp_label] += 1
            bagDescriptors[key] = bagDescriptor
        else:
            bagDescriptors[key].beans[temp_label] += 1

    print_to_arff("out.arff", bagDescriptors.values(), allClasses)

    print "hello"