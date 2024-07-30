import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.error import DependencyNotInstalled
import torch


class MultiValleyMountainCarEnv(gym.Env):
    """
    Description:
        The agent (a car) is started in one of outermost the bottom of valleys. For any given
        state the agent may choose to accelerate to the left, right.

    Source:
        The environment appeared developed based on Gym problem: MountainCar
        (https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/mountain_car.py).

    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -0.99          0.99
        1      Car Velocity              -0.07          0.07

    Actions:
        Type: Discrete(2)
        Num    Action
        0      Accelerate to the Left
        1      Accelerate to the Right

        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.

    Reward:
         Reward of 1 is awarded if the agent reached the area designated by the flags (|position| <= 0.05)
         on top of the mountain.
         Reward of 0 is awarded if the agent outside (|position| >= 0.05).

    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.67 , -0.77] or  [0.67 , 0.77] with equal probability
         The starting velocity of the car is assigned a uniform random value in
         [-0.01, 0.01].

    Episode Termination:
         Never
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, force = 0.001, dt = 0.1):
        self.min_position = -0.99
        self.max_position = 0.99
        self.min_height = -0.2
        self.max_height = 0.6
        self.dt = dt

        self.min_start_position = 0.67
        self.max_start_position = 0.77
        self.max_start_velocity = 0.01

        self.max_speed = 0.07

        self.min_goal_position = -0.05
        self.max_goal_position =  0.05

        self.force = force
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)


        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        position, velocity = self.state
        velocity += self._total_horizontal_force(torch.tensor(action),torch.tensor(position)).item() * self.dt
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity * self.dt
        position = np.clip(position, self.min_position, self.max_position)

        reward = 1 if self._is_on_traget(position) else 0

        self.state = (position, velocity)
        return np.array(self.state, dtype=np.float32), reward, False, {}

    def _is_on_traget(self,position):
         return (self.min_goal_position <= position) & (position <= self.max_goal_position)

    def reset(self):
        sign = np.random.randint(0,2)*2-1
        position = sign * self.np_random.uniform(low=self.min_start_position, high=self.max_start_position)
        velocity = self.np_random.uniform(low=-self.max_start_velocity, high=self.max_start_velocity)
        self.state = np.array([position, velocity])

        return np.array(self.state, dtype=np.float32)

    @staticmethod
    def _height(x):
        return 1/10 * (np.cos(2 * np.pi * x) + 2 * np.cos(4 * np.pi * x) - np.log(1 - x ** 2))

    @staticmethod
    def _tg_alpha(x):
        return 1/10 * (2 * x / (1 - x ** 2) - 2 * torch.pi * torch.sin(2 * torch.pi * x) - 8 * torch.pi * torch.sin(4 * torch.pi * x))

    def _total_horizontal_force(self,action,position):
        tg_alpha = self._tg_alpha(position)
        return (action * 2 - 1)  * self.force + tg_alpha * (-self.gravity)


    def render(self, mode="human"):
        
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e
        

        screen_width = 600 * 2
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40 / 2
        carheight = 20 / 2
        if self.screen is None:
            import pygame
            from pygame import gfxdraw

            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        # Draw the terrain
        xs = np.linspace(self.min_position, self.max_position, 100 * 2)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, (ys - self.min_height) * scale))
        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10 / 2

        # Draw the car
        pos = self.state[0]
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(np.arctan(self._tg_alpha(torch.tensor(pos))))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + (self._height(pos) - self.min_height) * scale,
                )
            )
        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        # Draw the wheels
        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(np.arctan(self._tg_alpha(torch.tensor(pos))))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + (self._height(pos) - self.min_height) * scale),
            )
            gfxdraw.aacircle(self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128))
            gfxdraw.filled_circle(self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128))

        # Draw the left flag
        flagx = int((self.min_goal_position - self.min_position) * scale)
        flagy1 = int((self._height(self.min_goal_position) - self.min_height) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))
        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        # Draw the right flag
        flagx = int((self.max_goal_position - self.min_position) * scale)
        flagy1 = int((self._height(self.max_goal_position) - self.min_height) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))
        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        # Update the screen
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(60)  # Assuming 60 FPS
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


    def reward(self, state, action, state_dot):
        position, velocity = torch.moveaxis(state, -1, 0)
        reward = torch.where(self._is_on_traget(position), 1, 0)
        return reward


    # Functions below are extension for Umbrella Reinforce



    def state_dot(self, state, action):
        position, velocity = torch.moveaxis(state, -1, 0)

        acceleration = self._total_horizontal_force(action,position)

        position_dot = velocity
        velocity_dot = acceleration

        state_dot = torch.moveaxis(torch.cat((position_dot.unsqueeze(0), velocity_dot.unsqueeze(0)),axis = 0),0,-1)

        return state_dot

    def start_prob(self,state, device = None):

        low = torch.tensor([self.min_start_position, - self.max_start_velocity],device = device)
        high = torch.tensor([self.max_start_position,self.max_start_velocity],device = device)
        delta = high - low
        p_inside = (1 / 2) * (1 / torch.prod(delta))

        is_inside_right = torch.all((low[np.newaxis, :] < state) * (state < high[np.newaxis, :]), axis=-1)
        is_inside_left = torch.all((-high[np.newaxis, :] < state) * (state < -low[np.newaxis, :]), axis=-1)
        p = torch.where(is_inside_left | is_inside_right , p_inside, 0)

        return p

    def divergrnce(self,state,action):
        return 0

    def rate(self,state,action, device = None):
        p0 = self.start_prob(state,device)
        v = self.state_dot(state,action)
        r = self.reward(state,action,v)
        div_v = self.divergrnce(state,action)
        return p0, r, v, div_v

    def reflecting_representation(self,state):

        x, v = torch.moveaxis(state, -1, 0)
        device = state.device

        x_bounds = torch.tensor([self.min_position, self.max_position]).to(device)
        v_bounds = torch.tensor([-self.max_speed, self.max_speed ]).to(device)

        boundary = x_bounds.expand((1,) * len(x.shape) + (2,))
        v_cos = torch.cos((v - v_bounds.min()) / (v_bounds.max() - v_bounds.min()) * torch.pi)
        x, v_cos = x.unsqueeze(-1), v_cos.unsqueeze(-1)
        distance_to_boundary, _ = torch.abs(x - boundary).min(axis=-1)

        return torch.cat((x, v_cos ** 2, v_cos * distance_to_boundary.unsqueeze(-1)), axis=-1)


class StandUpEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, dt = 0.1, torque = None):
        self.dt = dt
        self.gravity = 0.025
        self.torque = torque if torque is not None else 2 * self.gravity

        self.max_speed = 0.07

        self.delta_pi = np.pi/24

        self.phi1_range = [0,np.pi]
        self.phi2_range = [-np.pi, np.pi]

        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.array([0,-np.pi]), np.array([np.pi,np.pi]), dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        m1, m2 = self.m1_m2(action)

        phi1_dot, phi2_dot = self._angular_velocity(torch.tensor(self.phi1), torch.tensor(self.phi2), m1, m2)


        self.phi1 += phi1_dot.item() * self.dt
        self.phi2 += phi2_dot.item() * self.dt


        self.phi1 = np.clip(self.phi1, 0, np.pi)

        self.phi2 = np.clip(self.phi2, max(-2 * self.phi1,-np.pi), min(2 * (np.pi - self.phi1),np.pi))
        self.state = (self.phi1, self.phi2)
        reward = 1 if self._is_on_traget(*torch.tensor(self.state)) else 0
        return np.array(self.state, dtype=np.float32), reward, False, {}


    def _is_on_traget(self,phi1, phi2):
         return (torch.abs(phi1 - np.pi/2) <= self.delta_pi) & (torch.abs(phi2) <= self.delta_pi)

    def reset(self):
        right = np.random.rand() < (1/2)
        if right:
            self.phi1 = np.random.rand() * self.delta_pi
            self.phi2 = np.random.rand() * self.delta_pi
        else:
            self.phi1 = np.pi - np.random.rand() * self.delta_pi
            self.phi2 =       - np.random.rand() * self.delta_pi

        self.state = (self.phi1, self.phi2)
        return np.array(self.state, dtype=np.float32)


    def _angular_velocity(self,phi1, phi2, m1, m2):
        phi1_dot = m1 - m2 - self.gravity * torch.cos(phi1)
        phi2_dot =      m2 - self.gravity * torch.cos(phi1 + phi2)
        return phi1_dot, phi2_dot


    def render(self, mode="human"):

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e


        phi1, phi2 = self.state
        screen_width, screen_height = 500, 500
        bound = 2 + 0.2
        scale = screen_width / (2 * bound)
        center = (screen_width // 2, screen_height // 2)

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        def transform_to_screen(x, y):
            return (center[0] + int(x * scale), center[1] - int(y * scale))

        p1 = (0, 0)
        p2 = (np.cos(phi1), np.sin(phi1))
        p3 = (p2[0] + np.cos(phi1 + phi2), p2[1] + np.sin(phi1 + phi2))
        xys = [p1, p2, p3]
        thetas = [phi1, phi1 + phi2]

        # Draw ground line
        pygame.draw.line(self.surf, (0, 0, 0), (0, center[1]), (screen_width, center[1]))

        # Draw gray lines
        gray = (204, 204, 204)
        delta = self.delta_pi
        p_upper = (2 * np.cos(np.pi / 2 + delta), 2 * np.sin(np.pi / 2 + delta))
        p_lower = (2 * np.cos(np.pi / 2 - delta), 2 * np.sin(np.pi / 2 - delta))
        pygame.draw.line(self.surf, gray, center, transform_to_screen(*p_upper))
        pygame.draw.line(self.surf, gray, center, transform_to_screen(*p_lower))

        for (x, y), th in zip(xys, thetas):
            l, r, t, b = 0, 1, 0.05, -0.05
            vertices = [
                (l, b), (l, t), (r, t), (r, b)
            ]
            vertices = [pygame.math.Vector2(v).rotate_rad(th) for v in vertices]
            vertices = [(v[0] + x, v[1] + y) for v in vertices]
            vertices = [transform_to_screen(v[0], v[1]) for v in vertices]

            gfxdraw.aapolygon(self.surf, vertices, (0, 204, 204))
            gfxdraw.filled_polygon(self.surf, vertices, (0, 204, 204))

            circ_pos = transform_to_screen(x, y)
            gfxdraw.aacircle(self.surf, circ_pos[0], circ_pos[1], int(0.05 * scale), (204, 204, 0))
            gfxdraw.filled_circle(self.surf, circ_pos[0], circ_pos[1], int(0.05 * scale), (204, 204, 0))

        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(60)  # Assuming 60 FPS
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def reward(self, state, action, state_dot):
        phi1, phi2 = torch.moveaxis(state, -1, 0)
        reward = torch.where(self._is_on_traget(phi1, phi2) , 1, 0)
        return reward

    def m1_m2(self,action):
        m1 = (2 * (action // 2) - 1) * self.torque
        m2 = (2 * (action %  2) - 1) * self.torque
        return m1,m2


    # Functions below are extension for Umbrella Reinforce


    def state_dot(self, state, action):

        phi1, phi2 = torch.moveaxis(state, -1, 0)
        m1, m2 = self.m1_m2(action)
        phi1_dot, phi2_dot = self._angular_velocity(phi1,phi2,m1,m2)



        state_dot = torch.cat((phi1_dot.unsqueeze(-1), phi2_dot.unsqueeze(-1)),axis = -1)

        return state_dot

    def start_prob(self,state, device = None):
        phi1, phi2 = torch.moveaxis(state, -1, 0)

        is_inside_right = (                    phi1 < self.delta_pi) & (             0 <= phi2) & (phi2 < self.delta_pi)
        is_inside_left  = (torch.pi - self.delta_pi < phi1)          & (-self.delta_pi <= phi2) & (phi2 < 0)
        p_inside = 1/(2 * self.delta_pi ** 2)

        p = torch.where(is_inside_left | is_inside_right , p_inside, 0)

        return p

    @staticmethod
    def valid_state(state):
        phi1, phi2 = torch.moveaxis(state, -1, 0)
        phi1_in_range = (0 < phi1) & (phi1 <= np.pi)
        phi2_in_range = (-np.pi < phi2) & (phi2 <= np.pi)

        omega = phi1 + phi2/2
        height_in_range = (0 < omega) &  (omega <= np.pi)
        state_is_valid =  phi1_in_range & phi2_in_range & height_in_range
        return state_is_valid

    def divergrnce(self,state,action):
        phi1, phi2 = torch.moveaxis(state, -1, 0)
        return self.gravity * (torch.sin(phi1) + torch.sin(phi1 + phi2))

    def rate(self,state,action, device = None):
        p0 = self.start_prob(state,device)
        v = self.state_dot(state,action)
        r = self.reward(state,action,v)
        div_v = self.divergrnce(state,action)
        return p0, r, v, div_v


    def reflecting_representation(self,state):
        phi1, phi2 = torch.moveaxis(state, -1, 0)
        theta1 = phi1 - np.pi/2 # theta1 in (-pi/2,pi/2)
        scale = 2 * (1 - torch.abs(theta1/np.pi)) # scale in (1,2)
        theta2 = (phi2 + theta1)/scale   # theta2_ in (-pi/2,pi/2)

        h1 = torch.sin(theta1)
        h2 = torch.sin(theta2)

        return torch.cat((h1.unsqueeze(-1), h2.unsqueeze(-1)), axis=-1)


