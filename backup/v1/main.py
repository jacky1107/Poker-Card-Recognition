import os
import cv2
import json
import math
import numpy as np
import matplotlib.pyplot as plt

from config import *
from data import getData

name = 0
path = getData()
currentPath = os.getcwd()
dataFolder = os.path.join(currentPath, path["data"])
trainPath = os.path.join(dataFolder, path["train"])
trainData = os.listdir(trainPath)

testPath = os.path.join(dataFolder, path["test"])
testData = os.listdir(testPath)

#==============main==============#
for testName in range(len(testData)):
    images = os.path.join(testPath, testData[testName])
    
    img_clr = cv2.imread(images)
    img_clr = cv2.resize(img_clr, (1000, 1000))
    img_gray = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)
    hi, wi, c = img_clr.shape

    thresMin, thresMax = 170, 255
    img_blur = cv2.medianBlur(img_clr, 5)
    b, g, r = cv2.split(img_blur)
    img_b = cv2.threshold(b, thresMin, thresMax, cv2.THRESH_BINARY)[1]
    img_g = cv2.threshold(g, thresMin, thresMax, cv2.THRESH_BINARY)[1]
    img_r = cv2.threshold(r, thresMin, thresMax, cv2.THRESH_BINARY)[1]

    img_b = img_b.astype("int32")
    img_g = img_g.astype("int32")
    img_r = img_r.astype("int32")

    img_new = (img_b + img_g + img_r) / 3
    img_new = img_new.astype("uint8")
    img_new = cv2.threshold(img_new, thresMin, thresMax, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5,5))
    img_dilate = cv2.dilate(img_new, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)

    roiPokers = []
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_erode)
    lblareas = stats[:, cv2.CC_STAT_AREA]
    for j in range(1, len(lblareas)):
        scaler = 6
        x = stats[j, cv2.CC_STAT_LEFT] - scaler
        y = stats[j, cv2.CC_STAT_TOP] - scaler
        w = stats[j, cv2.CC_STAT_WIDTH] + scaler * 2
        h = stats[j, cv2.CC_STAT_HEIGHT] + scaler * 2
        hi, wi, ci = img_clr.shape
        if x < 0: x = 0
        if y < 0: y = 0
        img_copy = np.copy(img_erode[y:y+h,x:x+w])
        temp = np.copy(img_clr[y:y+h,x:x+w])

        area = lblareas[j]
        b, g, r = cv2.split(temp)
        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        mean_gray = calcMean(hist_gray)

        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
        mean_r = calcMean(hist_r)

        if area > 1000 and area < 45000: # 45000 is wrong
            img_sob = sobel(temp)
            img_thres = cv2.threshold(img_sob, 127, 255, cv2.THRESH_BINARY)[1]
            thres_area = np.sum(img_thres/255)
            if thres_area > 2000:
                temp_blur = cv2.GaussianBlur(temp, (5,5), 0)
                img_sob = sobel(temp_blur)
                gray_sob = cv2.cvtColor(img_sob, cv2.COLOR_BGR2GRAY)
                gray_thres = cv2.threshold(gray_sob, 64, 255, cv2.THRESH_BINARY)[1]

                img_copy_thres = cv2.threshold(img_copy, 127, 255, cv2.THRESH_BINARY_INV)[1]

                kernel = np.ones((5,5))
                img_copy_mor = cv2.morphologyEx(img_copy_thres, cv2.MORPH_OPEN, kernel, iterations=2)
                _, _, statsCopy, _ = cv2.connectedComponentsWithStats(img_copy_mor)
                copyLblareas = statsCopy[:, cv2.CC_STAT_AREA]
                for k in range(2, len(copyLblareas)):
                    xx = statsCopy[k, cv2.CC_STAT_LEFT]
                    yy = statsCopy[k, cv2.CC_STAT_TOP]
                    ww = statsCopy[k, cv2.CC_STAT_WIDTH]
                    hh = statsCopy[k, cv2.CC_STAT_HEIGHT]
                    img_copy[yy:yy+hh, xx:xx+ww] = 255

                img_copy = cv2.morphologyEx(img_copy, cv2.MORPH_CLOSE, kernel, iterations=2)

                ch, cw = img_copy.shape
                cah = ch // 2
                caw = cw // 2

                leftTop = img_copy[:cah,:caw]
                leftBot = img_copy[cah:,:caw]
                rightTop = img_copy[:cah,caw:]
                rightBot = img_copy[cah:,caw:]

                xx, yy = findCorner(leftTop, ksize=3, direction="leftTop")
                cv2.circle(temp, (xx, yy), 3, (255, 0, 0), -1)
                # img_res_leftTop = findCorner(leftTop, ksize=3, direction="leftTop")
                # img_res_leftBot = findCorner(leftBot, ksize=3, direction="leftBot")
                # img_res_rightTop = findCorner(rightTop, ksize=3, direction="rightTop")
                # img_res_rightBot = findCorner(rightBot, ksize=3, direction="rightBot")
                # img_res = np.zeros_like(img_copy)
                # img_res[:cah,:caw] = img_res_leftTop
                # img_res[cah:,:caw] = img_res_leftBot
                # img_res[:cah,caw:] = img_res_rightTop
                # img_res[cah:,caw:] = img_res_rightBot

                # print("image/" + str(name) + ".png")
                # cv2.imwrite("image/" + str(name) + ".png", img_res)
                name += 1

                # H_rows, W_cols, c = img_clr.shape
                # pts1 = np.float32([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
                # pts2 = np.float32([[0, 0], [W_cols,0], [0, H_rows], [H_rows, W_cols]])

                # M = cv2.getPerspectiveTransform(pts1, pts2)
                # dst = cv2.warpPerspective(img_clr, M, (1000,1000))

                cv2.imshow("temp", temp)
                cv2.waitKey(0)
cv2.destroyAllWindows()

"""

        kernel = np.ones((5,5))
        img_dilate = cv2.dilate(gray_thres, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.array(contours[0])
        print(contours)

        minAreaRect = cv2.minAreaRect(contours)
        rectCnt = np.int64(cv2.boxPoints(minAreaRect))
        print(rectCnt)
        leftTop = rectCnt[1]
        rightTop = rectCnt[2]
        leftBottom = rectCnt[0]
        rightBottom = rectCnt[3]

        cv2.circle(temp, (leftTop[0], leftTop[1]), 3, (255, 0, 0), -1)
        cv2.circle(temp, (rightTop[0], rightTop[1]), 3, (255, 0, 0), -1)
        cv2.circle(temp, (leftBottom[0], leftBottom[1]), 3, (255, 0, 0), -1)
        cv2.circle(temp, (rightBottom[0], rightBottom[1]), 3, (255, 0, 0), -1)




#         img_new = np.copy(img_clr)
#         img_test = np.zeros_like(img_clr)
#         contours, hierarchy = cv2.findContours(img_res_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         for i in range(len(contours)):
#             new = []
#             new_contours = np.zeros_like(contours[i])
#             if len(contours[i]) > 30:
#                 new_contours[:,0,0] = contours[i][:,0,0] + x
#                 new_contours[:,0,1] = contours[i][:,0,1] + y

#                 new.append(new_contours)
#                 result.append(new)

#                 minAreaRect = cv2.minAreaRect(new_contours)
#                 rectCnt = np.int64(cv2.boxPoints(minAreaRect))
#                 # (x, y, w, h) = cv2.boundingRect(new_contours)
#                 # rectCnt = np.array([[x, y + h], [x, y], [x + w, y], [x + w, y + h]])
#                 boxs.append(rectCnt)

# dsts = []
# for i in range(len(result)):
#     contours = result[i]
#     M = cv2.moments(contours[-1])
#     cx = int(M["m10"] / M["m00"])
#     cy = int(M["m01"] / M["m00"])

#     # cv2.circle(img_test, (cx, cy), 3, (255,0,255), -1)
#     cv2.drawContours(img_new, [boxs[i]], -1, (0, 0, 255), 1)
#     cv2.drawContours(img_test, contours, -1, (255,255,255), 1)

#     leftTop = boxs[i][1]
#     rightTop = boxs[i][2]
#     leftBottom = boxs[i][0]
#     rightBottom = boxs[i][3]
    
#     rx, ry = leftTop[0], leftTop[1]
#     width = int(abs(leftTop[0] - rightTop[0]))
#     height = int(abs(leftTop[1] - leftBottom[1]))

#     H_rows, W_cols, c = img_clr.shape
#     pts1 = np.float32([boxs[i][1], boxs[i][2], boxs[i][0], boxs[i][3]])
#     pts2 = np.float32([[0, 0], [W_cols,0], [0, H_rows], [H_rows, W_cols]])

#     M = cv2.getPerspectiveTransform(pts1, pts2)
#     dst = cv2.warpPerspective(img_clr, M, (450,450))
#     dsts.append(dst)

#     matches = 0
#     classification = ""
#     for trainName in range(len(trainData)):
#         train = os.path.join(trainPath, trainData[trainName])
#         img_train_clr = cv2.imread(train)
#         img_train_clr = cv2.resize(img_train_clr, (450, 450))
#         img_train_gray = cv2.cvtColor(img_train_clr, cv2.COLOR_BGR2GRAY)
#         img_train_blur = cv2.GaussianBlur(img_train_gray, (3,3), 0)
#         dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#         matchPoints = orbMatch(dst_gray, img_train_blur)
#         name = trainData[trainName].split(".")[0]
#         if len(matchPoints) > matches:
#             matches = len(matchPoints)
#             classification = trainData[trainName].split(".")[0]
#     print(classification, matches)
#     cv2.imshow("dst", dst)
#     cv2.waitKey(0)
#     print("\n")
# for i in range(len(dsts)):
#     cv2.imshow(str(i), dsts[i])
#     cv2.waitKey(0)
#     print("\n")

                # h, w = gray_thres.shape
                # yy = []
                # for y in range(h):
                #     yy.append(np.sum(gray_thres[y,:])/255)

                # xx = []
                # for x in range(w):
                #     xx.append(np.sum(gray_thres[:,x])/255)
                
                # plt.plot(np.arange(len(xx)), xx)
                # plt.plot(np.arange(len(yy)), yy)
                # plt.show()

"""
