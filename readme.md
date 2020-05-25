# Card Recognition

author: Jacky Wang

All code is written in Python, if there is any question please feel free to ask.

# Implementation

### dependencies
Please make sure you have installed those packages.
```
cv2
numpy
shutil
```

### Run program:
```
python3 main.py
```

# Preprocessing

1. Resize Image to (1000, 1000) to make sure that all input data have same size.
2. Use the sobel filter to detect the edge.
3. Remove noise from threshold the image, but this method will return edge with broken lines (Because it is hard to decide the threshold). In order to solve that problem, I use Morphology on the image.
4. Morphology can close or open the line, so that, I can find the completed edge with background infomation.
- Morphology has two operations
```
Opening: dilation after erosion
Closing: erosion  after dilation
```
After doing the preprocessing, I can get the following images.
![image](https://github.com/jacky1107/cardRecognition/blob/master/README_Image/compare.png)

# Region Of Interesting

After I get the closing image, I calculate the mean of each block.
That means, if I get higher mean, then it shows that this block has higher possibility is card.
After filter the image, I can get the following image.
![image](https://github.com/jacky1107/cardRecognition/blob/master/README_Image/regionOfInteresting.png)

After that, I need to get the points in each block in order to do [Perspective Transfromation](https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/)

There is the results of Perspective Transfromation below.
![image](https://github.com/jacky1107/cardRecognition/blob/master/README_Image/perspectiveTransform.png)

# Prediction
Now, I have every single card that has being transformed.
Next, I use orb detection to match 52 pokers (Ground Truth) and compare which matched points have maximum values.
Furthermore, in order to improve performance, I calculate mean of histogram of red channel and area of space/clubs/hearts/diamonds. After that, I can classify space/clubs or hearts/diamonds or JQK.

The results show below.
![image](https://github.com/jacky1107/cardRecognition/blob/master/README_Image/result1.png)
![image](https://github.com/jacky1107/cardRecognition/blob/master/README_Image/result2.png)

# Conclusion and Evaluation

The model predicts four cards successfully and failed two cards, and that the failed cards are Clubs J and Diamonds Q. Although precict failed, features of J and Q are very close, and predict diamond successfully.
And I think that the following are failed reasons: image too blur, difference of color too big.
而預測失敗的卡是梅花J預測成黑桃Q、菱形5預測成菱形10，雖然預測錯誤，但JQ的特徵算是非常接近，也成功預測到菱形。
我認為預測失敗的原因有以下幾項：圖像太模糊、顏色差異太大等等。