import rclpy
from rclpy.node import Node
# Float32MultiArray
from std_msgs.msg import Float32MultiArray
import spidev
import struct
from threading import Lock

def to_byte_list(data):
    tx_bytes = struct.pack('ffff', *data)
    return list(tx_bytes)

def from_byte_list(data):
    rx_bytes = bytes(data)
    return list(struct.unpack('ffff', rx_bytes))

class Stm32Bridge(Node):
    def __init__(self):
        super().__init__('stm32bridge')
        self.get_logger().info('STM32 Bridge Node has been started')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'motor_commands',
            self.motor_commands_callback,
            10)
        self.publisher = self.create_publisher(
            Float32MultiArray,
            'encoder_values',
            10)
        self.spi_lock = Lock()
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 100000
        self.last_encoder_published_time = self.get_clock().now()
        self.encoder_publish_timer = self.create_timer(0.1, self.publish_encoder_values_callback)

    def publish_encoder_values(self, byte_list):
        msg = Float32MultiArray()
        encoder_values = from_byte_list(byte_list)
        msg.data = encoder_values
        self.last_encoder_published_time = self.get_clock().now()
        self.publisher.publish(msg)

    def motor_commands_callback(self, msg):
        cmd_list = [float(x) for x in msg.data][:4] # Limit to 4 commands
        tx_byte_list = to_byte_list(cmd_list)
        with self.spi_lock:
            rx_byte_list = self.spi.xfer2(tx_byte_list)
        self.publish_encoder_values(rx_byte_list)

    def publish_encoder_values_callback(self):
        now = self.get_clock().now()
        time_since_last_publish = (now - self.last_encoder_published_time).nanoseconds / 1e9
        if time_since_last_publish > 0.1:
            null_cmd_list = [-1.0, -1.0, -1.0, -1.0]  # Dummy command to trigger encoder read
            null_cmd_byte_list = to_byte_list(null_cmd_list)
            with self.spi_lock:
                rx_byte_list = self.spi.xfer2(null_cmd_byte_list)
            self.publish_encoder_values(rx_byte_list)

def main(args=None):
    rclpy.init(args=args)

    stm32_bridge = Stm32Bridge()

    rclpy.spin(stm32_bridge)

    stm32_bridge.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
