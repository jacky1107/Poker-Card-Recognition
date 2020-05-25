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
![image](https://github.com/jacky1107/cardRecognition/blob/master/readmeImage/compare.png)

# Region Of Interesting

After I get the closing image, I calculate the mean of each block.
That means, if I get higher mean, then it shows that this block has higher possibility is card.
After filter the image, I can get the following image.
![image](https://github.com/jacky1107/cardRecognition/blob/master/readmeImage/regionOfInteresting.png)

After that, I need to get the points in each block in order to do [Perspective Transfromation](https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/)

There is the results of Perspective Transfromation below.
![image](https://github.com/jacky1107/cardRecognition/blob/master/readmeImage/perspectiveTransform.png)

# Prediction

# Evaluation

# Reference

(a) what other sources you used apart from the lecture material used in class during your work on the assignment
(b) how to compile and run your program
(c) any interesting features and extensions of your assignment.

Add a file evaluation.pdf
    which shows the results of the requested evaluations?

    Were the results expected or did they surprise you?
    
    Did the results highlight shortcomings in the system?
    
    Do you have any ideas for how to fix those? 
