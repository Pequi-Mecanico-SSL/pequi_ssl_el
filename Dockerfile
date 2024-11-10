FROM osrf/ros:iron-desktop
# FROM arm64v8/ros:iron-ros-base - use this for raspberry pi
SHELL ["/bin/bash", "-c"]

WORKDIR /ssl_ws

RUN apt-get update &&\
    apt-get install -y python3-pip nano git &&\
    apt-get remove -y modemmanager

RUN pip3 install mavproxy

WORKDIR /tmp

# Install uXRCE_DDS agent
RUN echo "Installing uXRCE_DDS agent..." && \
    git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git && \
    cd Micro-XRCE-DDS-Agent && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig /usr/local/lib/

WORKDIR /ssl_ws

COPY src src/

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

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
