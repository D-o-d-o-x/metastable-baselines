from gym import spaces
import numpy as np
import pygame


class Observable():
    def __init__(self):
        self.obs = None
        pass

    def get_observation_space():
        print("[!] Using dummyObservable. Env won't output anything")
        return spaces.Box(low=0, high=255,
                          shape=(1,), dtype=np.uint8)


class CnnObservable(Observable):
    def __init__(self, in_width=256, in_height=256, out_width=32, out_height=32, draw_width=128, draw_height=128, smooth_scaling=True):
        super(CnnObservable, self).__init__()
        self.in_width = in_width
        self.in_height = in_height
        self.out_width = out_width
        self.out_height = out_height
        self.draw_width = draw_width
        self.draw_height = draw_height
        if smooth_scaling:
            self.scaler = pygame.transform.smoothscale
        else:
            self.scaler = pygame.transform.scale

    def _set_env(self, env):
        self.env = env

    def get_observation_space(self):
        return spaces.Box(low=0, high=255,
                          shape=(self.out_width, self.out_height), dtype=np.uint8)

    def get_observation(self):
        x, y = self.env.agent.pos[0]*self.env.width - self.in_width / \
            2, self.env.agent.pos[1]*self.env.height - self.in_height/2
        w, h = self.in_width, self.in_height
        cx, cy = _clip(x, 0, self.env.width), _clip(
            y, 0, self.env.height)
        cw, ch = _clip(w, 0, self.env.width - cx), _clip(h,
                                                         0, self.env.height - cy)
        rect = pygame.Rect(cx, cy, cw, ch)
        snap = self.env.surface.subsurface(rect)
        self.snap = pygame.Surface((self.in_width, self.in_height))
        pygame.draw.rect(self.snap, (50, 50, 50),
                         pygame.Rect(0, 0, self.in_width, self.in_height))
        self.snap.blit(snap, (cx - x, cy - y))
        self.obs = self.scaler(
            self.snap, (self.out_width, self.out_height))
        return self.obs

    def draw(self):
        if not self.obs:
            self.get_observation()
        big = pygame.transform.scale(
            self.obs, (self.draw_width, self.draw_height))
        x, y = self.env.width - self.draw_width - 10, 10
        pygame.draw.rect(self.env.screen, (50, 50, 50),
                         pygame.Rect(x - 1, y - 1, self.draw_width + 2, self.draw_height + 2))
        self.env.screen.blit(
            big, (x, y))


def _clip(num, lower, upper):
    return min(max(num, lower), upper)
