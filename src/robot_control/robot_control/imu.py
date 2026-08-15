import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import struct
from threading import Lock
from smbus2 import SMBus
import math

class IMUNode(Node):
    def __init__(self):
        super().__init__('imu')
        self.get_logger().info('IMU Node has been started')
        self.publisher = self.create_publisher(
            Imu,
            'imu',
            10)
        
        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('publish_rate_hz', 200.0)
        self.i2cbus = SMBus(self.get_parameter('i2c_bus').value)
        self.gyro_acc_address = 0x6B
        self.compass_address = 0x1E
        
        self.registers = {
            'CTRL1_XL': 0x10,
            'CTRL2_G': 0x11,
            'CTRL3_C': 0x12,
            'CTRL9_XL': 0x18,
            'CTRL10_C': 0x19,
            'OUTX_L_G': 0x22,
            'OUTX_L_XL': 0x28,
        }
        self.max_gyro = (2000.0 * (math.pi) / 180.0) # rad/s
        self.gyro_scale = 1.16
        self.max_accel = 4.0 * 9.80665 # m/s^2
        self.max_int16 = 32768.0

        # gyroscope on all axis
        self.i2cbus.write_byte_data(self.gyro_acc_address, self.registers['CTRL10_C'], 0x38)
        # gyroscore on high performance mode, 1.66kHz, 2000dps max
        self.i2cbus.write_byte_data(self.gyro_acc_address, self.registers['CTRL2_G'], 0x7C)

        # accelerometer on all axis
        self.i2cbus.write_byte_data(self.gyro_acc_address, self.registers['CTRL3_C'], 0x38)
        # accelerometer on high performance mode, 1.66kHz, 4g max
        self.i2cbus.write_byte_data(self.gyro_acc_address, self.registers['CTRL1_XL'], 0x88)

        # enable BDU
        self.i2cbus.write_byte_data(self.gyro_acc_address, self.registers['CTRL3_C'], 0x44)

        rate = self.get_parameter('publish_rate_hz').value
        self.timer = self.create_timer(1.0 / rate, self.publish_imu)

    def publish_imu(self):
        msg = Imu()
        msg.header.frame_id = 'imu'
        msg.header.stamp = self.get_clock().now().to_msg()
        # gyro and accel output registers are contiguous: one 12-byte read
        raw = self.i2cbus.read_i2c_block_data(self.gyro_acc_address, self.registers['OUTX_L_G'], 12)
        data = struct.unpack('<hhhhhh', bytes(raw))
        gyro_data = [v * (self.max_gyro / self.max_int16) * self.gyro_scale for v in data[:3]]
        accel_data = [v * (self.max_accel / self.max_int16) for v in data[3:]]
        msg.angular_velocity.x = -gyro_data[0]
        msg.angular_velocity.y = -gyro_data[1]
        msg.angular_velocity.z = -gyro_data[2]
        # set linear acceleration
        msg.linear_acceleration.x = accel_data[0]
        msg.linear_acceleration.y = accel_data[1]
        msg.linear_acceleration.z = accel_data[2]
        
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    imu_node = IMUNode()

    rclpy.spin(imu_node)

    imu_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
