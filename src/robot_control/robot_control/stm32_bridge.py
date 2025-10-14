import rclpy
from rclpy.node import Node
# Float32MultiArray
from std_msgs.msg import Float32MultiArray
import struct
from threading import Lock
from smbus2 import SMBus, i2c_msg

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
        
        self.bus  = SMBus(1)
        self.addr = 0x42
        self.i2c_lock = Lock()
        self.latest_command = [50.0, 50.0, 50.0, 50.0]

        self.last_encoder_published_time = self.get_clock().now()
        self.encoder_publish_timer = self.create_timer(0.01, self.publish_encoder_values_callback)

    def publish_encoder_values(self, encoder_values):
        msg = Float32MultiArray()
        msg.data = encoder_values
        self.last_encoder_published_time = self.get_clock().now()
        self.publisher.publish(msg)
    
    def write_and_read(self, floats_out):
        with self.i2c_lock:
            # Write 16 bytes to the STM32
            tx_bytes = struct.pack('<4f', *floats_out)
            self.bus.i2c_rdwr(i2c_msg.write(self.addr, tx_bytes))
            self.latest_command = floats_out
            # Read 16 bytes back from the STM32
            rx = i2c_msg.read(self.addr, 16)
            self.bus.i2c_rdwr(rx)
            floats_in = struct.unpack('<4f', bytes(rx))
            return list(floats_in)

    def motor_commands_callback(self, msg):
        cmd_list = [float(x) for x in msg.data][:4] # Limit to 4 commands
        encoder_values = self.write_and_read(cmd_list)
        self.publish_encoder_values(encoder_values)

    def publish_encoder_values_callback(self):
        now = self.get_clock().now()
        time_since_last_publish = (now - self.last_encoder_published_time).nanoseconds / 1e9
        if time_since_last_publish > 0.1:
            encoder_values = self.write_and_read(self.latest_command)
            self.publish_encoder_values(encoder_values)

def main(args=None):
    rclpy.init(args=args)

    stm32_bridge = Stm32Bridge()

    rclpy.spin(stm32_bridge)

    stm32_bridge.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
