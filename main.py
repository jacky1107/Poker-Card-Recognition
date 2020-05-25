import os
import cv2
import copy
import shutil
import numpy as np

from config import *
from data import getData

dsts = []

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
    if "DS" in images: os.remove(images)
    img_clr = cv2.imread(images)
    img_clr = cv2.resize(img_clr, (1000, 1000))
    img_gray = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)
    hi, wi, c = img_clr.shape

    img_sob = sobel(img_gray)
    img_thres = cv2.threshold(img_sob, 127, 255, cv2.THRESH_BINARY)[1]
    
    kernel = np.ones((5,5))
    img_dilate = cv2.dilate(img_thres, kernel, iterations=3)
    img_erode = cv2.erode(img_dilate, kernel, iterations=3)

    boxs = []
    result = []
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_erode)
    lblareas = stats[:,cv2.CC_STAT_AREA]
    for j in range(1, len(lblareas)):
        if lblareas[j] > 1000:
            scaler = 0
            x = stats[j, cv2.CC_STAT_LEFT]  - scaler
            y = stats[j, cv2.CC_STAT_TOP]  - scaler
            w = stats[j, cv2.CC_STAT_WIDTH]  + scaler * 2
            h = stats[j, cv2.CC_STAT_HEIGHT]  + scaler * 2

            temp = np.copy(img_clr[y:y+h,x:x+w])

            b, g, r = cv2.split(temp)
            gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            temp_sob = sobel(gray)

            hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_sob = cv2.calcHist([temp_sob], [0], None, [256], [0, 256])

            mean = calcMean(hist_gray)
            mean_sob = calcMean(hist_sob)

            if mean > 100 and mean_sob > 10:
                gray_blur = cv2.medianBlur(gray, 5)
                gray_sob = sobel(gray_blur)
                gray_thres = cv2.threshold(gray_sob, 32, 255, cv2.THRESH_BINARY)[1]
                gray_thres_inv = cv2.threshold(gray_sob, 32, 255, cv2.THRESH_BINARY_INV)[1]
                
                hi, wi = gray_thres_inv.shape
                for yy in range(hi):
                    startX = findminMax(gray_thres_inv[yy, :], direction="start")
                    endX = findminMax(gray_thres_inv[yy, :], direction="end")
                    if startX != -1 and endX != -1:
                        gray_thres[yy, startX:endX] = 255
                
                card_point, contours = getCardCornerPoints(gray_thres)
                if "None" not in str(type(card_point)):
                    dst = perspectiveTransform(card_point, temp)
                    dst = cv2.resize(dst, (691, 1056))
                    dsts.append(dst)

# shutil.rmtree("predict")
# os.mkdir("predict")

# groundTruth = [
#     "6S", "JC", "6H" , "AS", "5D", "10D",
#     "JC", "AS", "6H", "5D", "6S", "10D",
#     ]

# ground = 0
# correct = 0
# wrong = 0
# predict = True
# check = False
# for dst in dsts:
#     dst_res = copy.deepcopy(dst)

#     red, jackQueenKing = pokerPredict(dst, groundTruth, ground)
#     dst_gray =  cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#     dst_thres = cv2.threshold(dst_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

#     cv2.imwrite("predict/" + str(ground) + "_truth" + ".jpg", dst_res)
#     if predict:
#         nameList = []
#         matchPointsList = []
#         for trainName in range(len(trainData)):
#             name = trainData[trainName].split(".")[0]
#             train = os.path.join(trainPath, trainData[trainName])
#             checkJQK = "J" in name or "Q" in name or "K" in name
#             checkRed = "H" in name or "D" in name
#             if jackQueenKing:
#                 if checkJQK:
#                     img_train_blur, nameList, matchPointsList = match(train, dst_res, nameList, matchPointsList, name)
#             elif red:
#                 if checkRed:
#                     img_train_blur, nameList, matchPointsList = match(train, dst_res, nameList, matchPointsList, name)
#             elif not checkJQK and not checkRed:
#                 img_train_blur, nameList, matchPointsList = match(train, dst_res, nameList, matchPointsList, name)

#         predictCard = printPrediction(nameList, matchPointsList, groundTruth, ground, red, jackQueenKing, check=check)

#         train = os.path.join(trainPath, predictCard + ".png")
#         img_train_clr = cv2.imread(train)
#         img_train_clr = img_train_clr * 0.8
#         img_train_clr = img_train_clr.astype("uint8")
#         cv2.imwrite("predict/" + str(ground) + "_predict_" + predictCard + ".jpg", img_train_clr)
#         print(f"Truth: {groundTruth[ground]}")
#         print(f"Predict: {predictCard}")
#         print("\n")
#         if predictCard == groundTruth[ground]: correct += 1
#         else: wrong += 1
#     ground += 1

# acc = correct / (correct + wrong)
# print(f"Correct: {correct}")
# print(f"Wrong: {wrong}")
# print(f"Accuracy: {acc}")