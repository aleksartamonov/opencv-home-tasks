import numpy as np
import cv2
#from skimage import morphology
#from skimage.morphology import watershed
for i in xrange(100, 187):
    filename = 'image_0000'
    filename = filename[0:-len(str(i))] + str(i)
    img = cv2.imread('data/'+filename+'.jpg')
    img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Z = img_Lab.reshape((-1, 3))
    row_len = img.shape[1]
    col_len = img.shape[0]

    # convert to np.float32
    Z = np.float32(Z)
    # M = np.zeros((len(Z),5))
    # for index in xrange(len(Z)):
    #     z_coord = np.float32(np.array([index/col_len*1e0/row_len, (index - col_len*(index/col_len))*1e0/col_len]))
    #     z_color = Z[index]
    #     feat = np.append(z_color*1e0/255,0*z_coord)
    #     M[index] = feat
    # M = np.float32(M)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 12
    ret, label, center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    #

    for j in xrange(center.shape[0]):
        center_col = center[j]
        not_green = center_col[1] > 123 or center_col[2] < 127 or center_col[2] > 190
        if not_green:
            center[j] = np.array([0, 128, 128])
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)
    bgr_res = cv2.cvtColor(res2, cv2.COLOR_LAB2BGR)
    cv2.imwrite('green/'+filename+'.jpg', bgr_res)
    gray = cv2.cvtColor(bgr_res, cv2.COLOR_BGR2GRAY)
    gray_mask = np.ones((res2.shape[0], res2.shape[1]))*255
    mask = np.ones(res2.shape)*255
    notgreen = gray < 10
    mask[notgreen] = 0
    gray_mask[notgreen] = 0
    kernel_er = np.ones((2, 2), np.uint8)
    fg = cv2.erode(gray_mask, None, iterations = 10)
    fg = np.uint8(fg)
    cv2.imwrite('eroded/'+filename+'.jpg', fg)
    bgt = cv2.dilate(gray_mask,None,iterations = 20)
    bgt = np.uint8(bgt)
    ret,bg = cv2.threshold(bgt, 1, 128, 1)
    marker = cv2.add(fg,bg)
    marker32 = np.int32(marker)
    cv2.watershed(img,marker32)
    m = cv2.convertScaleAbs(marker32)
    ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    res = cv2.bitwise_and(img,img,mask = thresh)
    cv2.imwrite('result/'+filename+'.jpg', res)
    # cv2.imshow('res'+str(i),res)
    # cv2.waitKey(0)
    #cv2.imshow('Lab', img_Lab)