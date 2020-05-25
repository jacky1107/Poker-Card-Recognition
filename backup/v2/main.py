import os
import cv2
import json
import math
import numpy as np

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
    img_clr = cv2.resize(img_clr, (450, 450))
    img_gray = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)
    hi, wi, c = img_clr.shape

    img_sob = sobel(img_gray)
    img_thres = cv2.threshold(img_sob, 127, 255, cv2.THRESH_BINARY)[1]
    
    kernel = np.ones((5,5))
    img_dilate = cv2.dilate(img_thres, kernel, iterations=3)
    img_erode = cv2.erode(img_dilate, kernel, iterations=3)

    boxs = []
    result = []
    img_res = np.zeros_like(img_clr)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_erode)
    lblareas = stats[:,cv2.CC_STAT_AREA]
    for j in range(1, len(lblareas)):
        x = stats[j, cv2.CC_STAT_LEFT]
        y = stats[j, cv2.CC_STAT_TOP]
        w = stats[j, cv2.CC_STAT_WIDTH]
        h = stats[j, cv2.CC_STAT_HEIGHT]

        temp = np.copy(img_clr[y:y+h,x:x+w])

        b, g, r = cv2.split(temp)
        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

        mean = np.mean(hist_gray)
        count = np.sum(hist_gray[170:])

        if (mean > 1.5 and mean < 110) and (count > 200):
            img_res_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            img_res_gray = cv2.medianBlur(img_res_gray, 5)
            img_res_sob = sobel(img_res_gray)
            img_res_thres = cv2.threshold(img_res_sob, 32, 255, cv2.THRESH_BINARY)[1]

            cv2.imshow("img_res_thres", img_res_thres)
            cv2.waitKey(0)

            img_new = np.copy(img_clr)
            img_test = np.zeros_like(img_clr)
            _, contours, hierarchy = cv2.findContours(img_res_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i in range(len(contours)):
                new = []
                new_contours = np.zeros_like(contours[i])
                if len(contours[i]) > 30:
                    new_contours[:,0,0] = contours[i][:,0,0] + x
                    new_contours[:,0,1] = contours[i][:,0,1] + y

                    new.append(new_contours)
                    result.append(new)

                    minAreaRect = cv2.minAreaRect(new_contours)
                    rectCnt = np.int64(cv2.boxPoints(minAreaRect))
                    # (x, y, w, h) = cv2.boundingRect(new_contours)
                    # rectCnt = np.array([[x, y + h], [x, y], [x + w, y], [x + w, y + h]])
                    boxs.append(rectCnt)

    dsts = []
    for i in range(len(result)):
        contours = result[i]
        M = cv2.moments(contours[-1])
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # cv2.circle(img_test, (cx, cy), 3, (255,0,255), -1)
        cv2.drawContours(img_new, [boxs[i]], -1, (0, 0, 255), 1)
        cv2.drawContours(img_test, contours, -1, (255,255,255), 1)

        # leftTop = boxs[i][1]
        # rightTop = boxs[i][2]
        # leftBottom = boxs[i][0]
        # rightBottom = boxs[i][3]
        
        # rx, ry = leftTop[0], leftTop[1]
        # width = int(abs(leftTop[0] - rightTop[0]))
        # height = int(abs(leftTop[1] - leftBottom[1]))

        # H_rows, W_cols, c = img_clr.shape
        # pts1 = np.float32([boxs[i][1], boxs[i][2], boxs[i][0], boxs[i][3]])
        # pts2 = np.float32([[0, 0], [W_cols,0], [0, H_rows], [H_rows, W_cols]])

        # M = cv2.getPerspectiveTransform(pts1, pts2)
        # dst = cv2.warpPerspective(img_clr, M, (450,450))
        # dsts.append(dst)

        # matches = 0
        # classification = ""
        # for trainName in range(len(trainData)):
        #     train = os.path.join(trainPath, trainData[trainName])
        #     img_train_clr = cv2.imread(train)
        #     img_train_clr = cv2.resize(img_train_clr, (450, 450))
        #     img_train_gray = cv2.cvtColor(img_train_clr, cv2.COLOR_BGR2GRAY)
        #     img_train_blur = cv2.GaussianBlur(img_train_gray, (5,5), 0)
        #     dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        #     matchPoints = orbMatch(dst_gray, img_train_blur)
        #     name = trainData[trainName].split(".")[0]
        #     if len(matchPoints) > matches:
        #         matches = len(matchPoints)
        #         classification = trainData[trainName].split(".")[0]
        # print(classification, matches)
        # cv2.imshow("dst", dst)
        # cv2.waitKey(0)
        # print("\n")

    # for i in range(len(dsts)):
    #     pokerPredict(dsts[i])
    #     cv2.imshow(str(i), dsts[i])
    #     cv2.waitKey(0)
    #     print("\n")
    cv2.imshow("img_test", img_test)
    cv2.waitKey(0)
cv2.destroyAllWindows()