# Card Recognition

author: Jacky Wang

All code is written in Python, if there is any question please feel free to ask.

# Implementation

## dependencies
Please make sure you have installed those packages.
```
cv2
numpy
shutil
```

## Run the program:
```
python3 main.py
```

# Preprocessing

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


# How to Compile the program

Please make sure you have installed those packages.
```
numpy
matplotlib
pickle
```

- troubleshooting: pickle is not available if python.version < 3.6

Running:
``
python main.py [-s] [save_img, default:True]
``
The saved image will save into output folder   
After running main.py, it will show these infomation below:  
1. The true path of the robot  
2. The output of the Kalman Filter for the path of the robot  
3. In a separate subwindow show the variance of the Kalman Filter against time  
4. In a separate subwindow show the error of the Kalman Filter path against time  
5. Average error  
6. variance of error  

# Object detection example using Kalman filter
Interesting features and extensions of Kalman filter.
After running the code, make sure you have installed this package.
```
pip install opencv-python
```
Running:
``
python find_targeting_kalman_filter.py
``

This example will detect object with color blue.  
And green point is predicted value of Kalman filter.  
Red point is true point which is centroid of object.

The result shows that the green point will track the red point as much as possible.  
# cardRecognition
# cardRecognition
