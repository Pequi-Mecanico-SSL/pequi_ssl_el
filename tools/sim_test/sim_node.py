"""rsim_ros SimulatorNode, unchanged, plus /simulator/state: the full rSim state
of each step with a step counter ([step, state...]), so the test harness can
stamp everything on the simulator's own time axis, and /clock (step * 16 ms)
so all nodes run on simulated time.

rSim computes velocities from the position change since the previous
get_state() call, so the state is not fetched a second time: the node's own
call is captured by a pass-through wrapper."""
import rclpy
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64MultiArray

from src.simulator import SimulatorNode   # the rsim_ros package installs as 'src'


class CapturingSim:
    def __init__(self, sim):
        self._sim = sim
        self.last_state = None

    def step(self, actions):
        return self._sim.step(actions)

    def get_state(self):
        self.last_state = self._sim.get_state()
        return self.last_state

    def __getattr__(self, name):
        return getattr(self._sim, name)


class StampedSimulator(SimulatorNode):
    def __init__(self):
        super().__init__()
        self.sim = CapturingSim(self.sim)
        self.state_pub = self.create_publisher(Float64MultiArray, '/simulator/state', 10)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.step = 0

    def update_state_and_publish(self):
        # simulated time: every other node runs with use_sim_time
        self.step += 1
        self.clock_pub.publish(Clock(clock=Time(nanoseconds=self.step * 16_000_000).to_msg()))
        super().update_state_and_publish()
        self.state_pub.publish(Float64MultiArray(data=[float(self.step)] + list(self.sim.last_state)))


def main():
    rclpy.init()
    node = StampedSimulator()
    try:
        rclpy.spin(node)
    except Exception:
        pass   # shutdown (context invalid / Ctrl-C)


if __name__ == '__main__':
    main()
