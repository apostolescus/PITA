# PITA
Python Intelligent Traffic Assistant

## TODO

## **DON'T FORGET BACKUP**

### [ ] Convert scripts for raspbian

### **Coral TPU:**
 - [x] test tflite model
 - [ ] convert yolov4 lite model

### **Lane detection:**
- [x] add lane detector to pipeline
- [x] perform basic lane detection
- [x] add switch to GUI 
- [ ] document improvements
- [ ] compare to robust lane detection
- [ ] integrate to alerter

### **GPS:**
* [x] mocking module
* [ ] calculate speed
* [ ] connect gps module


### **Camera:**
- [x] test with webCam
- [x] recording modes
- [x] allow distance measurment
- [ ] connect GPIO Camera
- [ ] connect external screen

### **Network:**
- [ ] label other images
- [ ] train network
- [ ] convert network

### **Others:**
- [x] GUI
- [x] Alert system
- [ ] send information to firebase
- [ ] allow user to modify fix size time

### **Docs:**
- [x] recording modes documentation
- [ ] alert system documentation
- [ ] distance measure documentation
- [ ] code cleaning
- [ ] sources organisation in directories

## Image Detector

This class performs both object detection and distance estimation.
In order to perform object detection, it requires a YOLO trained model (*.weights),
the config file (.cfg) and the name file (.names).
For distance estimation, it requires a csv formatted file, with the average sizes of the objects.

Average size of the object may include it's width or height.
The implementation allows for flexible distance estimation. 
You can specify to use the width, the height or the biggest/smallest from those.

How to format csv file?

CSV file should contain 3 collums.
The first one represents the object id(used in .names file). 
The match between the detected object and it's sizes is done using this id.
! First id should be 0 !

Second collumn represents the object characteristic to use:
0 - use width
1 - use height
2 - use the maximum value 
3 - use the minimum value 

Example:

Suppose we have a class of cars.
Cars can differ a lot in height from model to model and thus is pretty hard to find
a good distance estimator.
The difference is less noticeable at the width ( all cars should fit the same road), thus 
we can use the width as a value for distance estimation.
Futhermore, consider the position of your camera. If it's going to be 
a dashcam, you will have sa higher probability of detecting more times full car width than 
full car height.
Another point is that the camera will always faceing the back of the fron car no matter what.
Ex: car_id,1,car_width

On the other hand, suppose you want to detect a cell phone.
The cell phone can be in vaious positions compared to a car.
We can't know for shure if it is vertical or horizonta, thus we don't know which is it's true width or height.
We know that a phone's height is, in most of the cases, bigger than it's width. We can use this property in distance
estimation by telling the algorithm to use the biggest one for calculation. 

Ex: phone_id,2,phone_height

Third column represents the value for the property specify in the second collumn.

How is distance estimation done?
TODO


## Alerter

The alert class is responsable for alerting the user when he is in danger.
Alerting system can be configured based on vechicle type and current weather/road conditions.
This parametes affect directly the braking distance, because of the friction coefficients that variates.

The alert system uses the speed recived from GPS and calculates minimum distance for a full stop, based also on 
driver reaction time. 

If the stopping distance is lower than the safety stopping time plus a delay parameter, the driver will be alerted.

## VideoManagerWrapper

VideoManagerWrapper provides a high level, thread safe API for the low level VideoManager.

Mediates acces between GUI, the other classes and VideoManager.

## VideoManager

VideoManager is a singleton class that manages low level recording operations.
Important parameters in constructor:

#possible to be removed
* file_name :  represents default file name (it will be overwritted by set_name method)
* time : represents the maximum seconds that will be recorded 
* mu : measure unit, represents a measurement unit for *time* parameter; by default time is taken in seconds
* frame_size : represents image size ( depends by camera )
* FPS : represents the rate at which the video will be generated

**set_name** : will generate a new file for every start/stop pair
                File name will be by the form "year|month|day_hour|minutes"

**record** : if it is in fixed mode will record to circular buffer, else will write frame to file

**save** : writes and release the video writer


## Recording Modes

PITA offers posibility to work as a smart dash cam.
Depending on your preferences you can select from three recording modes:

1. Smart Recording
2. Fixed Sized
3. Permanent Mode

All of the modes will start after the user press the "Start" button.
The video will be saved when user stops the recording by pressing "Stop"
button.

### Smart Recording

Smart recording is the is the recommended recording mode. 
How does it work? 
When an object is detected the algorithm aproximates the distance to it.
When the distance drops under a certain safe distance ( calculated based on the current speed, user experience and reaction time)  the user will be alerted and the app will start to record.

This way, the camera only records when a possible incident is detected.
Keep in mind that given the network error it might not perform always as expected.
Thus, if you want to be sure you don't miss any important event it is recommended to use
one of the other two modes.

### Permanent Recording

Permanent recording captures and stores everything the camera sees after the "Start" button was pressed. It records until the user stops it.
If you consider long journey, it can consume a lot of memory.

### Fixed Size Recording

Fixed Size Recording combines both of the recording modes.
It optimisez storage space, but also provides a certain safety that an any
important event won't be missed.
User sets the length of the frame time he wants to record.
The camera will record the last X seconds/minutes (X being the set period).




