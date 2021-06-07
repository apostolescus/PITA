# PITA
Python Intelligent Traffic Assistant

## TODO

## **DON'T FORGET BACKUP**

### **Server:** 
- [x] send single image localhost basic connection
- [x] implement DETECT request
- [x] send multiple images localhost basic connection
- [x] add image detection 
- [x] implement UPDATE request
- [x] implement UPDATE backend 
- [x] add traffic alert 
- [x] add basic lane detection
- [x] perform detection on GPU
- [x] add encrypted communication using ceritificates
- [x] allow both CPU/GPU detection
- [x] added config file
- [x] clean code
- [x] add traffic light/sempahores warning

### **Client:**
- [x] establish localhost connection
- [x] implement capture-display pipeline
- [x] add warning sound
- [x] add encrypted communication using ceritificates
- [x] convert script for raspbian
- [x] add CONFIG file
- [x] clean code
- [x] add average capture-process-return time measurements
- [x] send polygone with first message
- [x] display last alert in GUI
- [x] display speed in GUI
- [x] add custom sound for each alert

### **Coral TPU:**
 - [x] test tflite model
 - ~~[ ] convert yolov4 lite model~~

### **Lane detection:**
- [x] add lane detector to pipeline
- [x] perform basic lane detection
- [x] add switch to GUI 
- [x] use average line for short periods of time
- [ ] compare to robust lane detection

### **GPS:**
* [x] mocking module
* [x] calculate speed
* [x] connect gps module

### **Camera:**
- [x] test with webCam
- [x] fix recording mode
- [x] permanent recording mode
- [x] implement distance measurment
- [x] smart recording mode
- [x] start smart recording by default
- [x] connect GPIO Camera
- [ ] connect external screen

### **Network:**
- [x] implement both tf and yolo detection
- [x] add detection for TF lite
- [x] label other images
- [x] perform detection on GPU
- [x] train network
- [ ] improve network performances

### **Alerter:**
- [x] alert only for vehicles on the same 
lane
- [x] add connection to Firebase
- [x] upload data in other thread
- [x] make noise when in danger
- [x] send data to firebase when danger
- [x] save in redundancy database

### **Mobile:**
- [x] create GUI
- [x] fetch data from firebase
- [x] display alerts on mobile

### **Other:**
- [x] add logguru
- [x] add alerts for other type of incidents
- [x] show notification on GUI
- [x] remove GUI button for update
- [x] add extra infos to GUI
- [x] formatting and clean code

### **Testing: **
- [x] measure time using uuids
- [x] test lane detection
- [x] test alert sound
- [x] test object detector GPU/CPU + tiny/normal
- [x] test alert system (firebase, custom notifications)
- [x] test update system 
- [x] test video recording modes
- [ ] test on raspberry

**Optional**:
- [x] improve lane detection
- [x] split gpu detection multithread
- [x] split lane/object detection multithread
- [ ] add advanced lane detection

### **Docs:**
- [x] recording modes documentation
- [x] sources organisation in directories
- [ ] docs converting weights to tf
- [ ] alert system documentation
- [ ] distance measure documentation
- [ ] lane detection


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



### Server-Client Latency

On a wifi from a mobile device the latency between server and client is 200 milisenconds.
This mean that the picture taken by the device travels for about 0.12 s to the server and the response for about 0.7 s back to the device. The difference is explained by the size of the picture send compared to the one of the bounding boxes arrays ( sever times bigger).
Given a server processing time by around: 0.2 seconds ( on a CPU with tiny-yolo), the average trip time would be around 0.4s.
Given the human reaction time of 0.2-0.25s , the total capture-process-react time is aroung 0.6 seconds.
If we go one step futher and we consider the spreading of 5G networks in the following years, the latency of the comunication could be reduce to 0.1s.

***For 4G, this is 200 milliseconds, not far off the 250 milliseconds it takes for humans to react to visual stimuli. The 5G latency rate is significantly lower: at just 1 millisecond.*** 

**ThalesGroup: "5G vs 4G: what’s the difference?"**

Futher, if we integrate our system directly with the car and reduce human latency, the total time could be around 0.2-0.25 s( on a CPU). If we use a GPU for image detection and a more powerful computer, the total time could go as small as **0.1-0.15 seconds**. Almost half as the human reaction time.
Until futher tehnical upgrades, the system can be used as an auxiliar preventive one, in case the person is not paying attention to the road.
