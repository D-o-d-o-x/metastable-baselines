import pygame
import math


class Entity(object):
    def __init__(self, env):
        self.env = env
        self.pos = (env.random(), env.random())
        self.speed = (0, 0)
        self.acc = (0, 0)
        self.drag = 0
        self.radius = 10
        self.col = (255, 255, 255)
        self.shape = 'circle'

    def physics_step(self):
        x, y = self.pos
        vx, vy = self.speed
        ax, ay = self.acc
        vx, vy = vx+ax*self.env.acc_fac,  vy+ay*self.env.acc_fac
        x, y = x+vx*self.env.speed_fac, y+vy*self.env.speed_fac
        if x > 1 or x < 0:
            x = min(max(x, 0), 1)
            vx = 0
        if y > 1 or y < 0:
            y = min(max(y, 0), 1)
            vy = 0
        self.speed = vx/(1+self.drag), vy/(1+self.drag)
        self.pos = x, y

    def controll_step(self):
        pass

    def step(self):
        self.controll_step()
        self.physics_step()

    def draw(self):
        x, y = self.pos
        pygame.draw.circle(self.env.surface, self.col,
                           (x*self.env.width, y*self.env.height), self.radius, width=0)

    def on_collision(self, other):
        pass

    def kill(self):
        self.env.kill_entity(self)


class Agent(Entity):
    def __init__(self, env):
        super(Agent, self).__init__(env)
        self.pos = (0.5, 0.5)
        self.col = (0, 0, 255)
        self.drag = self.env.agent_drag
        self.controll_type = self.env.controll_type

    def controll_step(self):
        self._read_input()
        self.env.check_collisions_for(self)

    def _read_input(self):
        if self.controll_type == 'SPEED':
            self.speed = self.env.inp[0] - 0.5, self.env.inp[1] - 0.5
        elif self.controll_type == 'ACC':
            self.acc = self.env.inp[0] - 0.5, self.env.inp[1] - 0.5
        else:
            raise Exception('Unsupported controll_type')


class Enemy(Entity):
    def __init__(self, env):
        super(Enemy, self).__init__(env)
        self.col = (255, 0, 0)
        self.damage = 10

    def on_collision(self, other):
        if isinstance(other, Agent):
            self.env.new_reward -= self.damage


class Barrier(Enemy):
    def __init__(self, env):
        super(Barrier, self).__init__(env)


class CircleBarrier(Barrier):
    def __init__(self, env):
        super(CircleBarrier, self).__init__(env)


class Chaser(Enemy):
    def __init__(self, env):
        super(Chaser, self).__init__(env)
        self.target = self.env.agent
        self.arrow_fak = 100
        self.lookahead = 0

    def _get_arrow(self):
        tx, ty = self.target.pos
        x, y = self.pos
        fx, fy = x + self.speed[0]*self.lookahead*self.env.speed_fac, y + \
            self.speed[1]*self.lookahead*self.env.speed_fac
        dx, dy = (tx-fx)*self.arrow_fak, (ty-fy)*self.arrow_fak
        return self.env._limit_to_unit_circle((dx, dy))


class WalkingChaser(Chaser):
    def __init__(self, env):
        super(WalkingChaser, self).__init__(env)
        self.col = (255, 0, 0)
        self.chase_speed = 0.45

    def controll_step(self):
        arrow = self._get_arrow()
        self.speed = arrow[0] * self.chase_speed, arrow[1] * self.chase_speed


class FlyingChaser(Chaser):
    def __init__(self, env):
        super(FlyingChaser, self).__init__(env)
        self.col = (255, 0, 0)
        self.chase_acc = 0.5
        self.arrow_fak = 5
        self.lookahead = 8 + env.random()*2

    def controll_step(self):
        arrow = self._get_arrow()
        self.acc = arrow[0] * self.chase_acc, arrow[1] * self.chase_acc


class Reward(Entity):
    def __init__(self, env):
        super(Reward, self).__init__(env)
        self.col = (0, 255, 0)
        self.avaible = True
        self.enforce_not_on_barrier = False
        self.reward = 1

    def on_collision(self, other):
        if isinstance(other, Agent):
            self.on_collect()
        elif isinstance(other, Barrier):
            self.on_barrier_collision()

    def on_collect(self):
        self.env.new_reward += self.reward

    def on_barrier_collision(self):
        if self.enforce_not_on_barrier:
            self.pos = (self.env.random(), self.env.random())
            self.env.check_collisions_for(self)


class OnceReward(Reward):
    def __init__(self, env):
        super(OnceReward, self).__init__(env)
        self.reward = 100

    def on_collect(self):
        self.env.new_abs_reward += self.reward
        self.kill()


class TeleportingReward(OnceReward):
    def __init__(self, env):
        super(TeleportingReward, self).__init__(env)
        self.enforce_not_on_barrier = True
        self.env.check_collisions_for(self)

    def on_collect(self):
        self.env.new_abs_reward += self.reward
        self.pos = (self.env.random(), self.env.random())
        self.env.check_collisions_for(self)


class TimeoutReward(OnceReward):
    def __init__(self, env):
        super(TimeoutReward, self).__init__(env)
        self.enforce_not_on_barrier = True
        self.env.check_collisions_for(self)
        self.timeout = 10

    def set_avaible(self, value):
        self.avaible = value
        if self.avaible:
            self.col = (0, 255, 0)
        else:
            self.col = (50, 100, 50)

    def on_collect(self):
        if self.avaible:
            self.env.new_abs_reward += self.reward
            self.set_avaible(False)
            self.env.timers.append((self.timeout, self.set_avaible, True))
