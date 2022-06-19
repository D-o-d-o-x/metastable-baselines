import gym
from gym import spaces
import numpy as np
import pygame
import random as random_dont_use
import math
from . import entities, observables


class ColumbusEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, observable=observables.Observable(), fps=60, env_seed=3.1):
        super(ColumbusEnv, self).__init__()
        self.action_space = spaces.Box(
            low=0, high=1, shape=(2,), dtype=np.float32)
        observable._set_env(self)
        self.observable = observable
        self.observation_space = self.observable.get_observation_space()
        self.title = 'Untitled'
        self.fps = fps
        self.env_seed = env_seed
        self.joystick_offset = (10, 10)
        self.surface = None
        self.screen = None
        self.width = 720
        self.height = 720
        self.start_pos = (0.5, 0.5)
        self.speed_fac = 0.01/fps*60
        self.acc_fac = 0.03/fps*60
        self.agent_drag = 0  # 0.01 is a good value
        self.controll_type = 'SPEED'  # one of SPEED, ACC
        self.limit_inp_to_unit_circle = True
        self.aux_reward_max = 0  # 0 = off
        self.aux_reward_discretize = 0  # 0 = dont discretize
        self.draw_observable = True
        self.draw_joystick = True

        self.rng = random_dont_use.Random()
        self.reset()

    def _seed(self, seed):
        self.rng.seed(seed)

    def random(self):
        return self.rng.random()

    def _ensure_surface(self):
        if not self.surface:
            self.surface = pygame.Surface((self.width, self.height))
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(self.title)

    def _limit_to_unit_circle(self, coords):
        l_sq = coords[0]**2 + coords[1]**2
        if l_sq > 1:
            l = math.sqrt(l_sq)
            coords = coords[0] / l, coords[1] / l
        return coords

    def _step_entities(self):
        for entity in self.entities:
            entity.step()

    def _step_timers(self):
        new_timers = []
        for time_left, func, arg in self.timers:
            time_left -= 1/self.fps
            if time_left < 0:
                func(arg)
            else:
                new_timers.append((time_left, func, arg))
        self.timers = new_timers

    def sq_dist(self, entity1, entity2):
        return (entity1.pos[0] - entity2.pos[0])**2 + (entity1.pos[1] - entity2.pos[1])**2

    def dist(self, entity1, entity2):
        return math.sqrt(self._sq_dist(entity1, entity2))

    def _get_aux_reward(self):
        aux_reward = 0
        for entity in self.entities:
            if isinstance(entity, entities.Reward):
                if entity.avaible:
                    reward = self.aux_reward_max / \
                        (1 + self.sq_dist(entity, self.agent))

                    if self.aux_reward_discretize:
                        reward = int(reward*self.aux_reward_discretize*2) / \
                            self.aux_reward_discretize / 2

                    aux_reward += reward
        return aux_reward

    def step(self, action):
        inp = action[0], action[1]
        if self.limit_inp_to_unit_circle:
            inp = self._limit_to_unit_circle(((inp[0]-0.5)*2, (inp[1]-0.5)*2))
            inp = (inp[0]+1)/2, (inp[1]+1)/2
        self.inp = inp
        self._step_timers()
        self._step_entities()
        observation = self.observable.get_observation()
        reward, self.new_reward, self.new_abs_reward = self.new_reward / \
            self.fps + self.new_abs_reward, 0, 0
        self.score += reward  # aux_reward does not count towards the score
        if self.aux_reward_max:
            reward += self._get_aux_reward()
        return observation, reward, 0, self.score
        return observation, reward, done, info

    def check_collisions_for(self, entity):
        for other in self.entities:
            if other != entity:
                if self._check_collision_between(entity, other):
                    entity.on_collision(other)
                    other.on_collision(entity)

    def _check_collision_between(self, e1, e2):
        shapes = [e1.shape, e2.shape]
        shapes.sort()
        if shapes == ['circle', 'circle']:
            sq_dist = ((e1.pos[0]-e2.pos[0])*self.width) ** 2 \
                + ((e1.pos[1]-e2.pos[1])*self.height)**2
            return sq_dist < (e1.radius + e2.radius)**2
        else:
            raise Exception(
                'Checking for collision between unsupported shapes: '+str(shapes))

    def kill_entity(self, target):
        newEntities = []
        for entity in self.entities:
            if target != entity:
                newEntities.append(entity)
            else:
                del target
                break
        self.entities = newEntities

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(18):
            enemy = entities.CircleBarrier(self)
            enemy.radius = self.random()*40+50
            self.entities.append(enemy)
        for i in range(3):
            enemy = entities.FlyingChaser(self)
            enemy.chase_acc = self.random()*0.4*0.3  # *0.6+0.5
            self.entities.append(enemy)
        for i in range(0):
            reward = entities.TimeoutReward(self)
            self.entities.append(reward)
        for i in range(1):
            reward = entities.TeleportingReward(self)
            self.entities.append(reward)

    def reset(self):
        pygame.init()
        self.inp = (0.5, 0.5)
        # will get rescaled acording to fps (=reward per second)
        self.new_reward = 0
        self.new_abs_reward = 0  # will not get rescaled. should be used for one-time rewards
        self.score = 0
        self.entities = []
        self.timers = []
        self.agent = entities.Agent(self)
        self.setup()
        self.entities.append(self.agent)  # add it last, will be drawn on top
        self._seed(self.env_seed)
        return 0
        return observation  # reward, done, info can't be included

    def _draw_entities(self):
        for entity in self.entities:
            entity.draw()

    def _draw_observable(self, forceDraw=False):
        if self.draw_observable or forceDraw:
            self.observable.draw()

    def _draw_joystick(self, forceDraw=False):
        if self.draw_joystick:
            x, y = self.inp
            pygame.draw.circle(self.screen, (100, 100, 100), (50 +
                                                              self.joystick_offset[0], 50+self.joystick_offset[1]), 50, width=1)
            pygame.draw.circle(self.screen, (100, 100, 100), (20+int(60*x) +
                                                              self.joystick_offset[0], 20+int(60*y)+self.joystick_offset[1]), 20, width=0)

    def render(self, mode='human'):
        self._ensure_surface()
        pygame.draw.rect(self.surface, (0, 0, 0),
                         pygame.Rect(0, 0, self.width, self.height))
        self._draw_entities()
        self.screen.blit(self.surface, (0, 0))
        self._draw_observable()
        self._draw_joystick()
        pygame.display.update()

    def close(self):
        pygame.display.quit()
        pygame.quit()


class ColumbusTest3_1(ColumbusEnv):
    def __init__(self):
        super(ColumbusEnv, self).__init__(observables.CnnObservable())
