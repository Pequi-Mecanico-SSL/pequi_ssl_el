Build with:
```
docker-compose build
```
And run simulator with:
```
docker-compose run simulator
```
To build for raspberry py 4, change base image in Dockerfile from:
```
FROM osrf/ros:iron-desktop
```
to:
```
FROM arm64v8/ros:iron-ros-base
```
And build with `docker-compose build`.

To test communication latency, run `docker-compose run ping` on the computer and `docker-compose run pong` on the raspberry.
 
 After building the docker file run it using:

'sudo docker-compose run -ti --env="DISPLAY" simulator '

To start the px4_ros_com node:

'ros2 launch px4_ros_com sensor_combined_listener.launch.py'
