import rclpy
from rclpy.node import Node

from std_msgs.msg import Int64


class PongNode(Node):

    def __init__(self):
        super().__init__('pong')
        self.subscription = self.create_subscription(
            Int64,
            'ping',
            self.ping_callback,
            10)
        self.publisher_ = self.create_publisher(Int64, 'pong', 10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info('Pong node is running')

    def ping_callback(self, msg):
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    pong_node = PongNode()

    rclpy.spin(pong_node)

    pong_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
