import numpy as np
from collections import namedtuple
import pickle
from pathlib import Path
from typing import Optional
from ament_index_python.packages import get_package_share_directory

class Sim2Real:

    def __init__(self,
        field_length=9.0,
        field_width=6.0,
        max_ep_length=30*40,
        deterministic: bool = True,
        seed: Optional[int] = None,
        ):
        self.field_length = field_length
        self.field_width = field_width
        self.xmin = -self.field_length/2
        self.xmax = self.field_length/2
        self.ymin = -self.field_width/2
        self.ymax = self.field_width/2
        self.penalty_length = 1.0
        self.max_pos = max(self.ymax, self.xmax + self.penalty_length)
        self.max_w = np.rad2deg(9.6)  # 9.6 rad/s = 92 rpm
        self.max_v = 3.5  # max linear velocity (m/s)

        self.default_players = 3
        self.obs_size = 77
        self.action_size = 4  # [v_x, v_y, v_theta, kick]
        self.n_stacks = 8
        self.max_ep_length = max_ep_length
        self.steps = 0
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

        self.stack_observations = {
            **{f"blue_{i}": np.zeros(self.n_stacks*self.obs_size) for i in range(self.default_players)},
            **{f"yellow_{i}": np.zeros(self.n_stacks*self.obs_size) for i in range(self.default_players)}
        }
        self.last_actions = {
            **{f"blue_{i}": np.zeros(self.action_size) for i in range(self.default_players)},
            **{f"yellow_{i}": np.zeros(self.action_size) for i in range(self.default_players)}
        }

        self.robot_template = namedtuple('Robot', ['x', 'y', 'theta'])
        self.goal_template = namedtuple('Goal', ['x', 'y'])
        self.ball_template = namedtuple('Ball', ['x', 'y'])

        # Load model weights once
        share_dir = get_package_share_directory('strategy')
        weights_path = Path(share_dir) / 'weights' / 'rl_weights.pkl'
        with open(weights_path, "rb") as f:
            self.weights: dict = pickle.load(f)

    def state_to_action(self, state: dict, convert: bool = True) -> dict:
        """
        Função que transforma o estado do ambiente em observações
        
        :Param: *state* estado do ambiente. dict[str, list[float]]
            Exemplo:
                state = {
                    "blue_0": [x, y, theta],
                    "blue_1": [x, y, theta],
                    "blue_2": [x, y, theta],
                    "yellow_0": [x, y, theta],
                    "yellow_1": [x, y, theta],
                    "yellow_2": [x, y, theta],
                    "ball": [x, y]
                }
        :Param: *convert* se False, as saídas das ações serão entre -1 e 1 (sem unidade) e globais. 
                Caso for True, as saidas serão locais para cada robô e os valores absolutos. Padrão é True.

                
        :Return: *actions* ações para cada robô. dict[str, list[float]]
            Exemplo:
                actions = {
                    "blue_0": [vel_x (m/s), vel_y (m/s),  vel_theta (graus/s), vel_x_kick (m/s)],
                    "blue_1": [vel_x (m/s), vel_y (m/s),  vel_theta (graus/s), vel_x_kick (m/s)],
                    "blue_2": [vel_x (m/s), vel_y (m/s),  vel_theta (graus/s), vel_x_kick (m/s)],
                    "yellow_0": [vel_x (m/s), vel_y (m/s),  vel_theta (graus/s), vel_x_kick (m/s)],
                    "yellow_1": [vel_x (m/s), vel_y (m/s),  vel_theta (graus/s), vel_x_kick (m/s)],
                    "yellow_2": [vel_x (m/s), vel_y (m/s),  vel_theta (graus/s), vel_x_kick (m/s)],
                }

        """
        # state to observation
        observations = self._state_to_observation(state)

        for i in range(self.default_players):
            self.stack_observations[f'blue_{i}'] = np.delete(self.stack_observations[f'blue_{i}'], range(self.obs_size))
            
            self.stack_observations[f'blue_{i}'] = np.concatenate([
                self.stack_observations[f'blue_{i}'], 
                observations[f'blue_{i}']
            ], axis=0, dtype=np.float64)
        
        for i in range(self.default_players):
            self.stack_observations[f'yellow_{i}'] = np.delete(self.stack_observations[f'yellow_{i}'], range(self.obs_size))
            
            self.stack_observations[f'yellow_{i}'] = np.concatenate([
                self.stack_observations[f'yellow_{i}'], 
                observations[f'yellow_{i}']
            ], axis=0, dtype=np.float64)

        # observation to action (using cached weights)
        weights: dict = self.weights

        actions_out = {}
        last_actions_normalized = {}
        for i, robot_name in enumerate(self.stack_observations.keys()):
            robot_obs = self.stack_observations[robot_name]

            h1 = np.tanh((robot_obs @ weights["_hidden_layers.0._model.0.weight"].T) + weights["_hidden_layers.0._model.0.bias"])
            h2 = np.tanh((h1 @ weights["_hidden_layers.1._model.0.weight"].T) + weights["_hidden_layers.1._model.0.bias"])
            h3 = np.tanh((h2 @ weights["_hidden_layers.2._model.0.weight"].T) + weights["_hidden_layers.2._model.0.bias"])
            a = (h3 @ weights["_logits._model.0.weight"].T) + weights["_logits._model.0.bias"]

            a = np.clip(a, np.log(1e-06), -np.log(1e-06))
            a = np.log(np.exp(a) + 1.0) + 1.0  # softplus + 1 to keep >1
            alpha, beta = np.split(a, 2)

            if self.deterministic:
                action = alpha / (alpha + beta)
            else:
                action = self.rng.beta(alpha, beta)
            action = action * 2 - 1  # to [-1, 1]

            signal = np.array([-1, 1, -1, 1]) if "yellow" in robot_name else np.array([1, 1, 1, 1])
            action_global_norm = signal * action

            last_actions_normalized[robot_name] = action_global_norm.tolist()

            if convert:
                angle = state[robot_name][2]
                actions_out[robot_name] = self._convert_actions(action_global_norm, angle).tolist()
            else:
                actions_out[robot_name] = action_global_norm.tolist()

        self.steps += 1
        # Keep last_actions as normalized, global actions for next observation stack
        self.last_actions = last_actions_normalized.copy()

        return actions_out
    
    def _state_to_observation(self, state: dict):
        observations = {}
        for i in range(self.default_players):
            robot = self.robot_template(*state[f"blue_{i}"])
            robot_action = self.last_actions[f'blue_{i}']
            allys = [self.robot_template(*state[f"blue_{j}"]) for j in range(self.default_players) if j != i]
            allys_actions = [self.last_actions[f'blue_{j}'] for j in range(self.default_players) if j != i]
            advs = [self.robot_template(*state[f"yellow_{j}"]) for j in range(self.default_players)]

            ball = self.ball_template(*state[f"ball"])

            goal_adv = self.goal_template(x=   0.2 + self.field_length/2, y=0)
            goal_ally = self.goal_template(x= -0.2 - self.field_length/2, y=0)

            robot_obs = self._robot_observation(robot, allys, advs, robot_action, allys_actions, ball, goal_adv, goal_ally)

            observations[f"blue_{i}"] = robot_obs


        for i in range(self.default_players):
            robot = self._inverted_robot(self.robot_template(*state[f'yellow_{i}']))
            robot_action = self.last_actions[f'yellow_{i}']
            allys = [self._inverted_robot(self.robot_template(*state[f'yellow_{j}'])) for j in range(self.default_players) if j != i]
            allys_actions = [self.last_actions[f'yellow_{j}'] for j in range(self.default_players) if j != i]
            advs = [self._inverted_robot(self.robot_template(*state[f'blue_{j}'])) for j in range(self.default_players)]

            ball = self.ball_template(x=-state[f'ball'][0], y=state[f'ball'][1])

            goal_adv = self.goal_template(x=  -(-0.2 - self.field_length/2), y=0)
            goal_ally = self.goal_template(x= -( 0.2 + self.field_length/2), y=0)

            robot_obs = self._robot_observation(robot, allys, advs, robot_action, allys_actions, ball, goal_adv, goal_ally)

            observations[f'yellow_{i}'] = robot_obs

        return observations

    def _robot_observation(self, robot, allys, adversaries, robot_action: np.ndarray, allys_actions: list, ball, goal_adv, goal_ally):

        positions = []
        orientations = []
        dists = []
        angles = []
        last_actions = np.array([robot_action] + allys_actions).flatten()

        x_b, y_b, *_ = self._get_pos(ball)
        sin_BG_al, cos_BG_al, theta_BG_al = self._get_2dots_angle_between(goal_ally, ball)
        sin_BG_ad, cos_BG_ad, theta_BG_ad = self._get_2dots_angle_between(goal_adv, ball)
        dist_BG_al = self._get_dist_between(ball, goal_ally)
        dist_BG_ad = self._get_dist_between(ball, goal_adv)

        x_r, y_r, sin_r, cos_r, theta_r  = self._get_pos(robot)
        sin_BR, cos_BR, theta_BR = self._get_2dots_angle_between(ball, robot)
        dist_BR = self._get_dist_between(ball, robot)

        positions.append([x_r, y_r])
        orientations.append([sin_r, cos_r, theta_r])
        #orientations.append([theta_r])
        dists.append([dist_BR, dist_BG_al, dist_BG_ad])
        angles.append([
            sin_BR, cos_BR, theta_BR, 
            sin_BG_al, cos_BG_al, theta_BG_al, 
            sin_BG_ad, cos_BG_ad, theta_BG_ad
        ])

        for ally in allys:
            x_al, y_al, sin_al, cos_al, theta_al = self._get_pos(ally)
            sin_AlR, cos_AlR, theta_AlR = self._get_2dots_angle_between(ally, robot)
            ally_dist = self._get_dist_between(ally, robot)
            positions.append([x_al, y_al])
            orientations.append([sin_al, cos_al, theta_al])
            #orientations.append([theta_al])
            dists.append([ally_dist])
            angles.append([sin_AlR, cos_AlR, theta_AlR])
        
        for i in range(self.default_players - len(allys) - 1):
            print("não é pra entrar aqui")
            x_al, y_al, sin_al, cos_al, theta_al = 0, 0, 0, 0, 0
            sin_AlR, cos_AlR, theta_AlR = 0, 0, 0
            ally_dist = 0
            positions.append([x_al, y_al])
            orientations.append([sin_al, cos_al, theta_al])
            # orientations.append([theta_al])
            dists.append([ally_dist])
            angles.append([sin_AlR, cos_AlR, theta_AlR])

        
        for adv in adversaries:
            x_adv, y_adv, sin_adv, cos_adv, theta_adv = self._get_pos(adv)
            sin_AdR, cos_AdR, theta_AdR = self._get_2dots_angle_between(adv, robot)
            adv_dist = self._get_dist_between(adv, robot)
            positions.append([x_adv, y_adv])
            orientations.append([sin_adv, cos_adv, theta_adv])
            #orientations.append([theta_adv])
            dists.append([adv_dist])
            angles.append([sin_AdR, cos_AdR, theta_AdR])

        for i in range(self.default_players - len(adversaries)):
            x_adv, y_adv, sin_adv, cos_adv, theta_adv = 0, 0, 0, 0, 0
            sin_AdR, cos_AdR, theta_AdR = 0, 0, 0
            adv_dist = 0
            positions.append([x_adv, y_adv])
            orientations.append([sin_adv, cos_adv, theta_adv])
            #orientations.append([theta_adv])
            dists.append([adv_dist])
            angles.append([sin_AdR, cos_AdR, theta_AdR])

        positions.append([x_b, y_b])

        positions = np.concatenate(positions)
        orientations = np.concatenate(orientations)
        dists = np.concatenate(dists)
        angles = np.concatenate(angles)
        time_left = [(self.max_ep_length - self.steps)/self.max_ep_length]

        #print(f"len_pos: {len(positions)} \t len_ori: {len(orientations)} \t len_dist: {len(dists)} \t len_ang: {len(angles)} \t len_last_act: {len(last_actions)} \t len_time_left: {len(time_left)}")
        robot_obs = np.concatenate([positions, orientations, dists, angles, last_actions, time_left], dtype=np.float64)
        # robot_obs = np.concatenate([positions, last_actions, time_left], dtype=np.float64)
        return robot_obs
    
    def _inverted_robot(self, robot):
        return self.robot_template(
            x=-robot.x, 
            y=robot.y, 
            theta= 180 - robot.theta if robot.theta < 180 else 540 - robot.theta
        )
    
    def _get_3dots_angle_between(self, obj1, obj2, obj3):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        f = lambda x: np.isnan(x).any() or np.isinf(x).any()
        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])
        p3 = np.array([obj3.x, obj3.y])

        vec1 = p1 - p2
        vec2 = p3 - p2

        cos_theta = np.dot(vec1, vec2)/ (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        return np.sin(theta), np.cos(theta), theta/np.pi

    def _get_2dots_angle_between(self, obj1, obj2):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])

        diff_vec = p1 - p2
        theta = np.arctan2(diff_vec[1], diff_vec[0])

        return np.sin(theta), np.cos(theta), theta/np.pi
    
    def _get_dist_between(self, obj1, obj2):
        """Retorna a distância formada pela reta que liga o obj1 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])

        diff_vec = p1 - p2
        
        max_dist = np.linalg.norm([self.xmax - self.xmin, self.ymax - self.ymin])
        dist = np.linalg.norm(diff_vec)

        return np.clip(dist / max_dist, 0, 1)

    def _get_pos(self, obj):

        x = np.clip(obj.x / self.max_pos, -1.2, 1.2)
        y = np.clip(obj.y / self.max_pos, -1.2, 1.2)
        
        theta = np.deg2rad(obj.theta) if hasattr(obj, 'theta') else None
        sin = np.sin(theta) if theta is not None else None
        cos = np.cos(theta) if theta is not None else None
        theta = np.arctan2(sin, cos)/np.pi if theta is not None else None
        #tan = np.tan(theta) if theta else None

        return x, y, sin, cos, theta
    
    def _convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local
        Angle in degrees"""
        angle_rad = np.deg2rad(angle)

        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        v_x, v_y = v_x*np.cos(angle_rad) + v_y*np.sin(angle_rad),\
            -v_x*np.sin(angle_rad) + v_y*np.cos(angle_rad)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x,v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c

        return np.array([v_x, v_y, v_theta, action[3]])


def main(args=None):
    # testar a classe no ambiente SSLMultiAgentEnv

    IA = Sim2Real(
        field_length=9.0,
        field_width=6.0,
        max_ep_length=30*40
    )

    state = {
        'blue_0': [-1.5, 0, 0],
        'blue_1': [-2.0, -1.0, 0],
        'blue_2': [-2.0, 1.0, 0],
        'yellow_0': [1.5, 0, 0],
        'yellow_1': [2.0, -1.0, 0],
        'yellow_2': [2.0, 1.0, 0],
        'ball': [0, 0]
    }

    actions = IA.state_to_action(state, convert=False)
    print("Iteration 1:",actions,"\n")

    actions = IA.state_to_action(state, convert=False)
    print("Iteration 2:",actions,"\n")

    actions = IA.state_to_action(state, convert=False)
    print("Iteration 3:",actions,"\n")

    #from rSoccer.rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv
    #import yaml
    #from rewards import DENSE_REWARDS, SPARSE_REWARDS
    #from rSoccer.rsoccer_gym.judges.ssl_judge import Judge


    #with open("config.yaml") as f:
    #    # use safe_load instead load
    #    file_configs = yaml.safe_load(f)

    #env_config = file_configs["env"]
    #env_config["match_time"] = 40
    #env_config["dense_rewards"] = DENSE_REWARDS
    #env_config["sparse_rewards"] = SPARSE_REWARDS
    #env_config["judge"] = Judge

    #env = SSLMultiAgentEnv(**env_config)
    #obs, *_ = env.reset()


    #IA = Sim2Real(
    #    field_length=9.0,
    #    field_width=6.0,
    #    max_ep_length=30*40
    #)

    #while True:
    #    done= {'__all__': False}
    #    truncated = {'__all__': False}
    #    while not done['__all__'] and not truncated['__all__']:

    #        state = {
    #            'blue_0': [env.frame.robots_blue[0].x, env.frame.robots_blue[0].y, env.frame.robots_blue[0].theta],
    #            'blue_1': [env.frame.robots_blue[1].x, env.frame.robots_blue[1].y, env.frame.robots_blue[1].theta],
    #            'blue_2': [env.frame.robots_blue[2].x, env.frame.robots_blue[2].y, env.frame.robots_blue[2].theta],
    #            'yellow_0': [env.frame.robots_yellow[0].x, env.frame.robots_yellow[0].y, env.frame.robots_yellow[0].theta],
    #            'yellow_1': [env.frame.robots_yellow[1].x, env.frame.robots_yellow[1].y, env.frame.robots_yellow[1].theta],
    #            'yellow_2': [env.frame.robots_yellow[2].x, env.frame.robots_yellow[2].y, env.frame.robots_yellow[2].theta],
    #            'ball': [env.frame.ball.x, env.frame.ball.y]
    #        }

    #        action = IA.state_to_action(state)
    #        action.update({f"yellow_{i}": [0, 0, 0, 0]  for i in range(env.n_robots_yellow)})

    #        obs, reward, done, truncated, info = env.step(action)
    #        # breakpoint()
    #        env.render()
    #        # print(env.judge_info)
    #        #input()
    #        #input("Pess Enter to continue...")
    #    env.reset()

if __name__ == "__main__":
    main()
