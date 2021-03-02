# PITA
Python Intelligent Traffic Assistant

## TODO
**DON'T FORGET BACKUP**

- [x] recording modes (smart, permanent, fixed)
- [ ] recording modes documentation
- [ ] train network
- [ ] lane detection
- [x] GUI
- [ ] GPS module integration
- [ ] external screen integration
- [ ] camera calibration
- [ ] code cleaning
- [x] alert system
- [ ] alert system documentation
- [x] distance measure
- [ ] distance measure documentation
- [ ] safe app closing
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




