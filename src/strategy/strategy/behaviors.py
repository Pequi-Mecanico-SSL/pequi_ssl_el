from typing import Dict, Tuple
import numpy as np


ActionMap = Dict[str, Tuple[float, float, float]]  # robot_name -> (vx, vy, w_deg)


def zero_actions(n_blue: int, n_yellow: int) -> ActionMap:
    return {
        **{f'blue_{i}': (0.0, 0.0, 0.0) for i in range(n_blue)},
        **{f'yellow_{i}': (0.0, 0.0, 0.0) for i in range(n_yellow)},
    }


class BaseBehavior:
    def __init__(self, n_blue: int, n_yellow: int, team: str):
        self.n_blue = n_blue
        self.n_yellow = n_yellow
        self.team = team  # 'blue' or 'yellow'

    def step(self, state: dict, aux: dict) -> ActionMap:
        raise NotImplementedError

    # Utilities
    @staticmethod
    def _p_controller(curr_xy, target_xy, kp: float = 1.5, vmax: float = 1.0):
        dx = target_xy[0] - curr_xy[0]
        dy = target_xy[1] - curr_xy[1]
        vx = np.clip(kp * dx, -vmax, vmax)
        vy = np.clip(kp * dy, -vmax, vmax)
        return vx, vy

    @staticmethod
    def _face_point(curr_theta_deg: float, curr_xy, target_xy, kp_deg: float = 120.0, wmax_deg: float = 180.0):
        ang = np.rad2deg(np.arctan2(target_xy[1] - curr_xy[1], target_xy[0] - curr_xy[0]))
        err = (ang - curr_theta_deg + 180.0) % 360.0 - 180.0
        w = np.clip(kp_deg * (err / 180.0), -wmax_deg, wmax_deg)
        return w


class HaltBehavior(BaseBehavior):
    def step(self, state: dict, aux: dict) -> ActionMap:
        return zero_actions(self.n_blue, self.n_yellow)


class StopBehavior(BaseBehavior):
    def __init__(self, n_blue: int, n_yellow: int, team: str, keepout_radius: float = 0.5):
        super().__init__(n_blue, n_yellow, team)
        self.keepout = keepout_radius

    def step(self, state: dict, aux: dict) -> ActionMap:
        actions = zero_actions(self.n_blue, self.n_yellow)
        bx, by = state['ball']
        # If any bot within keepout, move it outward slowly
        for color, n in [('blue', self.n_blue), ('yellow', self.n_yellow)]:
            for i in range(n):
                rx, ry, th = state[f'{color}_{i}']
                d = np.hypot(rx - bx, ry - by)
                if d < self.keepout and d > 1e-3:
                    # Move radially outward, face away from ball
                    dirx, diry = (rx - bx) / d, (ry - by) / d
                    target = (bx + dirx * self.keepout, by + diry * self.keepout)
                    vx, vy = self._p_controller((rx, ry), target, kp=0.8, vmax=0.4)
                    w = self._face_point(th, (rx, ry), (bx, by), kp_deg=60.0, wmax_deg=120.0)
                    actions[f'{color}_{i}'] = (vx, vy, w)
        return actions


class RLBehavior(BaseBehavior):
    def __init__(self, n_blue: int, n_yellow: int, team: str, rl_model):
        super().__init__(n_blue, n_yellow, team)
        self.rl = rl_model

    def step(self, state: dict, aux: dict) -> ActionMap:
        acts = self.rl.state_to_action(state, convert=True)
        return {k: (v[0], v[1], v[2]) for k, v in acts.items()}


class PrepareKickoffBehavior(BaseBehavior):
    def __init__(self, n_blue: int, n_yellow: int, team: str, kicking_team: str):
        super().__init__(n_blue, n_yellow, team)
        self.kicking_team = kicking_team

    def step(self, state: dict, aux: dict) -> ActionMap:
        actions = zero_actions(self.n_blue, self.n_yellow)
        bx, by = state['ball']
        # Decide attack direction: +x towards opponent by default
        attack_dir = aux.get('attack_dir', 1.0)
        # Simple formation: i=0 behind ball, others spread laterally
        if self.kicking_team in ('blue', 'yellow'):
            n = self.n_blue if self.kicking_team == 'blue' else self.n_yellow
            for i in range(n):
                rx, ry, th = state[f'{self.kicking_team}_{i}']
                if i == 0:
                    target = (bx - 0.5 * attack_dir, by)
                elif i == 1:
                    target = (bx - 1.0 * attack_dir, by + 0.7)
                else:
                    target = (bx - 1.0 * attack_dir, by - 0.7)
                vx, vy = self._p_controller((rx, ry), target, kp=0.8, vmax=0.6)
                w = self._face_point(th, (rx, ry), (bx, by))
                actions[f'{self.kicking_team}_{i}'] = (vx, vy, w)
        # Non-kicking team: stay stopped and respect keepout handled by StopBehavior if needed
        return actions


class BallPlacementBehavior(BaseBehavior):
    def __init__(self, n_blue: int, n_yellow: int, team: str, placing_team: str):
        super().__init__(n_blue, n_yellow, team)
        self.placing_team = placing_team

    def step(self, state: dict, aux: dict) -> ActionMap:
        actions = zero_actions(self.n_blue, self.n_yellow)
        dp = aux.get('designated_position')  # (x,y) in meters
        bx, by = state['ball']
        if dp is None:
            return actions
        # One robot (0) pushes ball towards dp, others hold
        color = self.placing_team
        n = self.n_blue if color == 'blue' else self.n_yellow
        if n == 0:
            return actions
        rx, ry, th = state[f'{color}_0']
        # Target behind ball towards dp
        dirx, diry = dp[0] - bx, dp[1] - by
        norm = np.hypot(dirx, diry) + 1e-6
        dirx, diry = dirx / norm, diry / norm
        approach = (bx - 0.15 * dirx, by - 0.15 * diry)
        vx, vy = self._p_controller((rx, ry), approach, kp=1.0, vmax=0.6)
        w = self._face_point(th, (rx, ry), (bx, by))
        actions[f'{color}_0'] = (vx, vy, w)
        # If close, push gently
        if np.hypot(rx - approach[0], ry - approach[1]) < 0.08:
            actions[f'{color}_0'] = (0.6 * dirx, 0.6 * diry, w)
        return actions


class DirectFreeBehavior(BaseBehavior):
    def __init__(self, n_blue: int, n_yellow: int, team: str, free_team: str):
        super().__init__(n_blue, n_yellow, team)
        self.free_team = free_team

    def step(self, state: dict, aux: dict) -> ActionMap:
        actions = zero_actions(self.n_blue, self.n_yellow)
        bx, by = state['ball']
        color = self.free_team
        n = self.n_blue if color == 'blue' else self.n_yellow
        for i in range(n):
            rx, ry, th = state[f'{color}_{i}']
            if i == 0:
                target = (bx - 0.3, by)
            elif i == 1:
                target = (bx - 0.8, by + 0.6)
            else:
                target = (bx - 0.8, by - 0.6)
            vx, vy = self._p_controller((rx, ry), target, kp=0.8, vmax=0.6)
            w = self._face_point(th, (rx, ry), (bx, by))
            actions[f'{color}_{i}'] = (vx, vy, w)
        return actions

