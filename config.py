import cv2
import numpy as np

def match(train, dst, nameList, matchPointsList, name):
    if "DS" in train: os.remove(train)
    img_train_clr = cv2.imread(train)
    h, w, c = img_train_clr.shape
    ha = h // 2
    wa = w // 2
    img_train_clr = img_train_clr * 0.8
    img_train_clr = img_train_clr.astype("uint8")
    img_train_gray = cv2.cvtColor(img_train_clr, cv2.COLOR_BGR2GRAY)
    img_train_blur = cv2.GaussianBlur(img_train_gray, (5,5), 0)
    matchPoints = orbMatch(dst, img_train_blur)
    nameList.append(name)
    matchPointsList.append(len(matchPoints))
    return img_train_blur, nameList, matchPointsList

def pokerPredict(dst, groundTruth, ground):
    h, w, c = dst.shape
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    dst_thres = cv2.threshold(dst_gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    for y in range(h):
        for x in range(w):
            if dst_thres[y, x] == 0:
                dst[y, x] = 0
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(hsv)

    red, jackQueenKing = False, False
    cr = cv2.threshold(cr, 127, 255, cv2.THRESH_BINARY_INV)[1]
    mean, std = cv2.meanStdDev(cr)
    area = np.sum(dst_thres) / 255
    if mean < 40: red = True
    elif area > 3e+5: jackQueenKing = True
    return red, jackQueenKing

def printPrediction(nameList, matchPointsList, groundTruth, ground, red, jackQueenKing, check=False):
    nameList = np.array(nameList)
    sortedList = np.argsort(matchPointsList)
    nameList = nameList[sortedList]
    matchPointsList = sorted(matchPointsList)
    if check:
        for i in range(len(matchPointsList)):
            printing(i, nameList, matchPointsList, groundTruth, ground)
        print("\n")
    return nameList[-1]

def printing(i, nameList, matchPointsList, groundTruth, ground):
    if nameList[i] == groundTruth[ground]:
        print("\n")
        print("\t", nameList[i], matchPointsList[i])
        print("\n")
    else:
        print(nameList[i], matchPointsList[i])

def findminMax(oneDarray, direction=""):
    if direction == "start":
        for x, pixel in enumerate(oneDarray):
            if pixel == 0:
                return x
    if direction == "end":
        for x in range(len(oneDarray)-1, 0, -1):
            if oneDarray[x] == 0:
                return x
    return -1

def getCardCornerPoints(edge, thres=0.05):
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    card_point = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        epsilon = thres * perimeter
        approximation = cv2.approxPolyDP(contour, epsilon, True)
        if len(approximation) == 4:
            card_point = approximation
            break
    return card_point, contours

def perspectiveTransform(card_point, image):
    point1 = card_point[0][0]
    point2 = card_point[1][0]
    point3 = card_point[2][0]
    point4 = card_point[3][0]

    card_point = [point1, point2, point3, point4]

    len12 = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    len23 = np.sqrt((point2[0] - point3[0])**2 + (point2[1] - point3[1])**2)
    len34 = np.sqrt((point3[0] - point4[0])**2 + (point3[1] - point4[1])**2)
    len41 = np.sqrt((point4[0] - point1[0])**2 + (point4[1] - point1[1])**2)

    len1234 = (len12 + len34) / 2
    len2341 = (len23 + len41) / 2

    if len1234 <= len2341:
        width = len1234
        height = len2341
        transform_point = [[width, 0], [0, 0], [0, height], [width, height]]

    elif len1234 > len2341:
        width = len2341
        height = len1234
        transform_point = [[0, 0], [0, height], [width, height], [width, 0]]

    src = np.float32(card_point)
    dst = np.float32(transform_point)

    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    card_img = cv2.warpPerspective(image, transform_matrix, (int(width), int(height)))

    return card_img

def sobel(image):
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    
def orbMatch(imgTestGray, imgTrainGray):
    orb = cv2.ORB_create()

    kpTrain = orb.detect(imgTrainGray,None)
    kpTrain, desTrain = orb.compute(imgTrainGray, kpTrain)

    kpCam = orb.detect(imgTestGray,None)
    kpCam, desCam = orb.compute(imgTestGray, kpCam)

    good = []
    if "None" not in str(type(desCam)) and "None" not in str(type(desTrain)):
        bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(np.asarray(desTrain, np.uint8), np.asarray(desCam, np.uint8), k=2)
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.8*n.distance:
                    good.append([m])
    return good

def calcMean(hists):
    s = 0
    m = np.sum(hists)
    for i, hist in enumerate(hists):
        s += (i * hist)
    return s / m
