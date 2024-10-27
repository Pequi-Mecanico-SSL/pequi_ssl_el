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
