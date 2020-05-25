# Card Recognition

author: Jacky Wang

All codes are written in Python. If you have any question please feel free to ask.

# Implementation

### dependencies
Please make sure you have installed the following packages.
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

1. Resize Image to (1000, 1000) to make sure that all input data has the same size.
2. Use the sobel filter to detect the edge.
3. Remove noise from the threshold image. However, this method will result in a return edge with broken lines (Because the threshold is hard to define). In order to solve this problem, I use Morphology on the image.
4. Morphology can close or open the line so that I can find the completed edge with background infomation.
- Morphology has two operations
```
Opening: erode and then dilate the image
Closing: dilate and then erode the image
```
After preprocessing, I got the following images.
![image](https://github.com/jacky1107/cardRecognition/blob/master/README_Image/compare.png)

# Region Of Interesting

After getting the closing image, I calculate the mean of each block.
If I get a higher mean, it means that the block has a higher possibility to be a card.
After filtering the image, this is the result.
![image](https://github.com/jacky1107/cardRecognition/blob/master/README_Image/regionOfInteresting.png)

Then, I need to get the points in each block in order to do [Perspective Transfromation](https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/)

Below is the result of Perspective Transfromation.
![image](https://github.com/jacky1107/cardRecognition/blob/master/README_Image/perspectiveTransform.png)

# Prediction
Now, every single card is transformed.
Next, I use ORB detection to campare the 52 pokers (Ground Truth) with the transformed image that has the highest match points.
Furthermore, in order to improve performance, I calculate the mean of histogram (red channel) and the area of each suit (space/clubs/hearts/diamonds). After that, I classify space and club, heart and diamond, and JQK as a group respectively.

Below are the results.
![image](https://github.com/jacky1107/cardRecognition/blob/master/README_Image/result1.png)
![image](https://github.com/jacky1107/cardRecognition/blob/master/README_Image/result2.png)

# Conclusion and Evaluation

The model successfully predicts four cards but failed Clubs J and Diamonds Q. Although it failed predicting the two, it identifies the similar image in the middle of J and Q. Also, it succeeds in predicting the suit (diamond).
I think there are severel reasons that leads to the defect:
1. The image is too blur
2. The difference between colors is too big.
3. The size of the numbers is too small 
4. Some numbers are broken

The above factors make the image hard to detect.