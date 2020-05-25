import cv2
import numpy as np

def findCorner(image, ksize, direction=""):
    h, w = image.shape
    maxValue = 0
    buffer = 2
    k = ksize // 2
    if direction == "leftTop":
        for y in range(buffer + k, h - buffer - k):
            for x in range(buffer + k, w - buffer - k):
                target = image[y-k:y+k+1, x-k:x+k+1]
                
                upperKernel = image[y-buffer-k:y-buffer+k+1, x-k:x+k+1]
                lowerKernel = image[y+buffer-k:y+buffer+k+1, x-k:x+k+1]
                rightKernel = image[y-k:y+k+1, x+buffer-k:x+buffer+k+1]
                leftKernel  = image[y-k:y+k+1, x-buffer-k:x-buffer+k+1]

                leftTopKernel = image[y-buffer-k:y-buffer+k+1, x-buffer-k:x-buffer+k+1]
                leftBotKernel = image[y+buffer-k:y+buffer+k+1, x-buffer-k:x-buffer+k+1]
                rightTopKernel = image[y-buffer-k:y-buffer+k+1, x+buffer-k:x+buffer+k+1]
                rightBotKernel  = image[y+buffer-k:y+buffer+k+1, x+buffer-k:x+buffer+k+1]

                s = 0
                s += np.sum(target - upperKernel)
                s += np.sum(target - lowerKernel)
                s += np.sum(target - rightKernel)
                s += np.sum(target - leftKernel)
                s += np.sum(target - leftTopKernel)
                s += np.sum(target - leftBotKernel)
                s += np.sum(target - rightTopKernel)
                s += np.sum(target - rightBotKernel)
                if s > maxValue:
                    maxValue = s
                    bestX = x
                    bestY = y
    return bestX, bestY
    # if direction == "leftBot":
    # if direction == "rightTop":
    # if direction == "rightBot":

def findPoint(gray_thres, direction="top"):
    h, w = gray_thres.shape
    point = 0
    total = 0
    average = 0
    if direction == "top":
        for y in range(int(h/2), -1, -1):
            count = np.sum(gray_thres[y, :])
            if count == 0:
                point = y + 1
                for x in range(len(gray_thres[point,:])):
                    if gray_thres[y+1,x] == 255:
                        average += x
                        total += 1
                break
    elif direction == "left":
        for x in range(int(w/2), -1, -1):
            count = np.sum(gray_thres[:, x])
            if count == 0:
                point = x + 1
                for y in range(len(gray_thres[:, point])):
                    if gray_thres[y,x+1] == 255:
                        average += y
                        total += 1
                break
    return average, total, point

def calcMean(hists):
    s = 0
    m = np.sum(hists)
    for i, hist in enumerate(hists):
        s += (i * hist)
    return s / m

def pokerPredict(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
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
                if m.distance < 0.75*n.distance:
                    good.append([m])
    return good

def findAngle(positions):
    coss = []
    start = positions[0]
    for i in range(1, len(positions)):
        end = positions[i]
        x1, y1 = start
        x2, y2 = end
        d1 = (x1 ** 2 + y1 ** 2) ** 0.5
        d2 = (x2 ** 2 + y2 ** 2) ** 0.5
        d = x1 * x2 + y1 * y2
        cos = d / (d1 + d2)
        coss.append(cos)
        start = end
    
    p = []
    cos = coss[0]
    for i in range(1, len(coss)):
        end = coss[i]
        diff = abs(cos - end)
        if diff > 10: p.append(positions[i])
        cos = end
    return p

def findContourPostion(temp):
    img_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    positions = []
    for y in range(h):
        for x in range(w):
            if img_gray[y, x] == 255:
                arr = np.array([x, y])
                positions.append(arr)
    return np.array(positions)

def find(positions, cx, cy, rx, ry):
    record_dist = 0
    size = len(positions)
    for i in range(size):
        x, y = positions[i]
        for j in range(size):
            wx, wy = positions[j]
            dist = (x - wx) ** 2 + (y - wy) ** 2
            if dist > record_dist:
                record_dist = dist
                x1 = x
                y1 = y
                x2 = wx
                y2 = wy
    return (x1 + rx, y1 + ry, x2 + rx, y2 + ry)

def findPoints(temp, cx, cy, rx, ry, direction="left"):
    newPoint = []
    img_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    if direction == "left":
        for x in range(cx, rx - 1, -1):
            pixel = img_gray[cy - ry, x - rx]
            if pixel > 200:
                newPoint.append([x, cy])
    if direction == "right":
        for x in range(cx, rx + w, 1):
            pixel = img_gray[cy - ry, x - rx]
            if pixel > 200:
                newPoint.append([x, cy])
    if direction == "top":
        for y in range(cy, ry - 1, -1):
            pixel = img_gray[y - ry, cx - rx]
            if pixel > 200:
                newPoint.append([cx, y])
    if direction == "bottom":
        for y in range(cy, ry + h, 1):
            pixel = img_gray[y - ry, cx - rx]
            if pixel > 200:
                newPoint.append([cx, y])
    return newPoint[-1]

def sobel(image):
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

def hough(img_clr, img_gray, minVotes=100):
    rhoAccuracy = 1
    thetaAccuracy = np.pi/180.0

    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLines(img_gray, rhoAccuracy, thetaAccuracy, minVotes)

    print(len(lines))
    scores = []
    positions = []
    if ( lines is not None ):
        for line in lines:
            for rho, theta in line:
                score = 0
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(img_clr,(x1,y1),(x2,y2),(255,0,0),1)
                # img_clr = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)

                # h, w = test.shape
                # for y in range(h):
                #     for x in range(w):
                #         if test[y, x] == img_gray[y, x]:
                #             score += 1
                # scores.append(score)
                # positions.append([x1, y1, x2, y2])
    # test = np.zeros_like(img_clr)
    # positions = np.array(positions)
    # sortScores = np.argsort(scores)[::-1]
    # positions = positions[sortScores]
    # for x1, y1, x2, y2 in positions[:10]:
    #     cv2.line(test,(x1,y1),(x2,y2),(255,255,255),1)
    return img_clr

def houghP(img_clr, img_gray, minVotes=10, minLineLength=10, maxLineGap=10, rhoAccuracy=1, thetaAccuracy=np.pi/180.0):
    dists = []
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(img_gray,rhoAccuracy,thetaAccuracy,minVotes,minLineLength,maxLineGap)
    if ( lines is not None ):
        for line in lines:
            for x1,y1,x2,y2 in line:
                r = np.random.randint(0, 255)
                g = np.random.randint(0, 255)
                b = np.random.randint(0, 255)
                dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dist > 1000:
                    cv2.line(img_clr,(x1,y1),(x2,y2),(r, g, b),5)
    return img_clr

#==================FIND_POINTS_TODO=================#
# kernel = np.ones((5,5))
# img_test[ry:ry + height, rx:rx + width] = cv2.dilate(img_test[ry:ry + height, rx:rx + width], kernel)

# temp = img_test[ry:ry + height, rx:rx + width]
# left = findPoints(temp, cx, cy, rx, ry, direction="left")
# right = findPoints(temp, cx, cy, rx, ry, direction="right")
# top = findPoints(temp, cx, cy, rx, ry, direction="top")
# bottom = findPoints(temp, cx, cy, rx, ry, direction="bottom")

# cv2.circle(img_test, (left[0], left[1]), 1, (255,255,255), -1)
# cv2.circle(img_test, (right[0], right[1]), 1, (255,255,255), -1)
# cv2.circle(img_test, (top[0], top[1]), 1, (255,255,255), -1)
# cv2.circle(img_test, (bottom[0], bottom[1]), 1, (255,255,255), -1)

# positions = findContourPostion(temp)
# p = findAngle(positions)

# for i in range(len(p)):
#     x, y = p[i]
#     cv2.circle(img_test, (x + rx, y + ry), 3, (0,0,255), -1)

# x1, y1 = left[0], top[1]
# x2, y2 = right[0], top[1]
# x3, y3 = left[0], bottom[1]
# x4, y4 = right[0], bottom[1]
# pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
