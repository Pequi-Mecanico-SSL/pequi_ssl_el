# ssl_vision_connector  
This ROS 2 package receives Protobuf messages from ssl-vision and converts them into ROS 2 messages  

### Build  
```bash
git clone https://github.com/Pequi-Mecanico-SSL/ssl_vision_connector
cd ssl_vision_connector
colcon build
```

### Launch  
```bash
source /opt/ros/iron/setup.bash
source install/setup.bash
ros2 launch ssl_vision_connector ssl_vision_connector_launch.yaml
```
