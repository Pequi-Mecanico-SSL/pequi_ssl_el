import rclpy
from rclpy.node import Node

from std_msgs.msg import Int64


class PingNode(Node):

    def __init__(self):
        super().__init__('ping')
        self.publisher_ = self.create_publisher(Int64, 'ping', 10)
        self.latencies = []
        # self.i = 0
        self.subscription = self.create_subscription(
            Int64,
            'pong',
            self.pong_callback,
            10)
        timer_period = 1/100.0  # seconds
        self.timer = self.create_timer(timer_period, self.send_ping_callback)
        self.get_logger().info('Ping Node Started')

    def send_ping_callback(self):
        msg = Int64()
        now_us = self.get_clock().now().nanoseconds / 1000.0
        msg.data = int(now_us)
        # msg.data = self.i
        # self.i += 1
        self.publisher_.publish(msg)
        self.get_logger().info('Sent ping at {}'.format(now_us))

    def pong_callback(self, msg):
        # print time difference
        now_us = self.get_clock().now().nanoseconds / 1000.0
        latency = now_us - msg.data
        latency_ms = latency / 1000.0
        self.latencies.append(latency_ms)
        avg_latency = sum(self.latencies) / len(self.latencies)
        self.get_logger().info('Latency: {:.2f}ms, Average: {:.2f}ms'.format(latency_ms, avg_latency))


def main(args=None):
    rclpy.init(args=args)

    ping_node = PingNode()

    rclpy.spin(ping_node)

    ping_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
