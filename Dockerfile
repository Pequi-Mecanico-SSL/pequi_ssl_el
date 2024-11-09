FROM osrf/ros:iron-desktop
# FROM arm64v8/ros:iron-ros-base - use this for raspberry pi
SHELL ["/bin/bash", "-c"]

WORKDIR /ssl_ws

RUN apt-get update &&\
    apt-get install -y python3-pip nano git

COPY src src/

WORKDIR src/

RUN git clone https://github.com/PX4/px4_msgs.git && \
    git clone https://github.com/PX4/px4_ros_com.git

WORKDIR /ssl_ws

RUN for d in src/*/ ; do \
    if [ -f "$d/requirements.txt" ]; then \
        echo "Installing requirements for $d"; \
        pip install -r $d/requirements.txt; \
    fi; \
done

RUN source /opt/ros/iron/setup.bash && \
    colcon build

COPY ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh

# Configuring boot firmware
RUN echo "Configuring boot firmware..." && \
    echo -e "\nenable_uart=1\ndtoverlay=disable-bt" >> /boot/firmware/config.txt && \
    echo "Checking for serial port availability..." && \
    cd / && \
    ls /dev/ttyAMA0 || echo "Serial port /dev/ttyAMA0 not found."

# Install MAVProxy and configure for PX4
RUN echo "Installing MAVProxy and setting up PX4 connection..." && \
    apt-get install -y python3-pip && \
    pip3 install mavproxy && \
    apt-get remove -y modemmanager && \
    echo "MAVProxy installed. Use the following command to start it:" && \
    echo "mavproxy.py --master=/dev/serial0 --baudrate 57600"

# Install uXRCE_DDS agent
RUN echo "Installing uXRCE_DDS agent..." && \
    git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git && \
    cd Micro-XRCE-DDS-Agent && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    ldconfig /usr/local/lib/

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
