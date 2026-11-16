import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.latest_msg = None
        self.timer = self.create_timer(3.0, self.timer_callback)

    def listener_callback(self, msg):
        # Called for every message; just remember the newest one
        self.latest_msg = msg

    def timer_callback(self):
        # Called every 3 seconds; print whatever arrived last
        if self.latest_msg is None:
            self.get_logger().info('No joint_states received yet')
            return
        msg = self.latest_msg
        joints = ', '.join(
            f'{name}={pos:.3f}' for name, pos in zip(msg.name, msg.position))
        self.get_logger().info(f'I heard: {joints}')


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()