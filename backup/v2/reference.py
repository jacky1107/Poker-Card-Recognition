
# def findAngle(positions):
#     coss = []
#     start = positions[0]
#     for i in range(1, len(positions)):
#         end = positions[i]
#         x1, y1 = start
#         x2, y2 = end
#         d1 = (x1 ** 2 + y1 ** 2) ** 0.5
#         d2 = (x2 ** 2 + y2 ** 2) ** 0.5
#         d = x1 * x2 + y1 * y2
#         cos = d / (d1 + d2)
#         coss.append(cos)
#         start = end
    
#     p = []
#     cos = coss[0]
#     for i in range(1, len(coss)):
#         end = coss[i]
#         diff = abs(cos - end)
#         if diff > 10: p.append(positions[i])
#         cos = end
#     return p

# def findContourPostion(temp):
#     img_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
#     h, w = img_gray.shape
#     positions = []
#     for y in range(h):
#         for x in range(w):
#             if img_gray[y, x] == 255:
#                 arr = np.array([x, y])
#                 positions.append(arr)
#     return np.array(positions)

# def find(positions, cx, cy, rx, ry):
#     record_dist = 0
#     size = len(positions)
#     for i in range(size):
#         x, y = positions[i]
#         for j in range(size):
#             wx, wy = positions[j]
#             dist = (x - wx) ** 2 + (y - wy) ** 2
#             if dist > record_dist:
#                 record_dist = dist
#                 x1 = x
#                 y1 = y
#                 x2 = wx
#                 y2 = wy
#     return (x1 + rx, y1 + ry, x2 + rx, y2 + ry)

# def findPoints(temp, cx, cy, rx, ry, direction="left"):
#     newPoint = []
#     img_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
#     h, w = img_gray.shape
#     if direction == "left":
#         for x in range(cx, rx - 1, -1):
#             pixel = img_gray[cy - ry, x - rx]
#             if pixel > 200:
#                 newPoint.append([x, cy])
#     if direction == "right":
#         for x in range(cx, rx + w, 1):
#             pixel = img_gray[cy - ry, x - rx]
#             if pixel > 200:
#                 newPoint.append([x, cy])
#     if direction == "top":
#         for y in range(cy, ry - 1, -1):
#             pixel = img_gray[y - ry, cx - rx]
#             if pixel > 200:
#                 newPoint.append([cx, y])
#     if direction == "bottom":
#         for y in range(cy, ry + h, 1):
#             pixel = img_gray[y - ry, cx - rx]
#             if pixel > 200:
#                 newPoint.append([cx, y])
#     return newPoint[-1]



# def hough(img_clr, img_gray, minVotes=100):
#     rhoAccuracy = 1
#     thetaAccuracy = np.pi/180.0

#     img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
#     lines = cv2.HoughLines(img_gray, rhoAccuracy, thetaAccuracy, minVotes)

#     print(len(lines))
#     scores = []
#     positions = []
#     if ( lines is not None ):
#         for line in lines:
#             for rho, theta in line:
#                 score = 0
#                 a = np.cos(theta)
#                 b = np.sin(theta)
#                 x0 = a*rho
#                 y0 = b*rho
#                 x1 = int(x0 + 1000*(-b))
#                 y1 = int(y0 + 1000*(a))
#                 x2 = int(x0 - 1000*(-b))
#                 y2 = int(y0 - 1000*(a))
#                 cv2.line(img_clr,(x1,y1),(x2,y2),(255,0,0),1)
#                 # img_clr = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)

#                 # h, w = test.shape
#                 # for y in range(h):
#                 #     for x in range(w):
#                 #         if test[y, x] == img_gray[y, x]:
#                 #             score += 1
#                 # scores.append(score)
#                 # positions.append([x1, y1, x2, y2])

#     # test = np.zeros_like(img_clr)
#     # positions = np.array(positions)
#     # sortScores = np.argsort(scores)[::-1]
#     # positions = positions[sortScores]
#     # for x1, y1, x2, y2 in positions[:10]:
#     #     cv2.line(test,(x1,y1),(x2,y2),(255,255,255),1)
#     return img_clr

# def houghP(img_clr, img_gray, minVotes=10, minLineLength=10, maxLineGap=10, rhoAccuracy=1, thetaAccuracy=np.pi/180.0):
#     dists = []
#     img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
#     lines = cv2.HoughLinesP(img_gray,rhoAccuracy,thetaAccuracy,minVotes,minLineLength,maxLineGap)
#     if ( lines is not None ):
#         for line in lines:
#             for x1,y1,x2,y2 in line:
#                 r = np.random.randint(0, 255)
#                 g = np.random.randint(0, 255)
#                 b = np.random.randint(0, 255)
#                 dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2)
#                 if dist > 1000:
#                     cv2.line(img_clr,(x1,y1),(x2,y2),(r, g, b),5)
#     return img_clr
