__author__ = 'aleksart'


import os
import cv2


def load_data(path):
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.jpg']
    images = []
    for file in result:
        images.append(file)
    return images




if __name__ == "__main__":
    data = load_data('../../../1/data')
    print len(data)


