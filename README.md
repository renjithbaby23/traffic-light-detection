# traffic-light-detection


Open-cv based approach demonstrating traffic signal detection (Red, Amber, Green, Unknown) on traffic images from a static camera.



### Requirements


Python 3.9.13


```
pip install numpy==1.23.4
pip install opencv-python==4.6.0
pip install matplotlib==3.6.1
```


### How To:


Execute the below command from terminal

```
python detect_signal.py --image sample_images/vlcsnap-2022-10-10-10h19m26s597.png --verbose 2
```

Note: To detect on new images coming from a different camera from a different angle, needs manual caliberation of the location of traffic light.


Sample Output:
```
Arguments:
Namespace(image='sample_images/vlcsnap-2022-10-10-10h19m26s597.png', hist_thresh=0.5, verbose=2)
scores (the lower the score, more similar): 
	red	 : 1.000
	amber	 : 1.000
	green	 : 0.000

Current Traffic Signal:  GREEN
```
