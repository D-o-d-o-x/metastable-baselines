from time import sleep, time
from env import ColumbusEnv, ColumbusTest3_1
import numpy as np
import pygame

from observables import Observable, CnnObservable


def main():
    env = ColumbusTest3_1()
    env.start_pos = [0.6, 0.3]
    playEnv(env)
    env.close()


def playEnv(env):
    env.reset()
    done = False
    while not done:
        t1 = time()
        env.render()
        pos = (0.5, 0.5)
        for event in pygame.event.get():
            pass
            # if event.type == pygame.MOUSEBUTTONDOWN:
            #    pos = pygame.mouse.get_pos()
            #    print(pos)
        pos = pygame.mouse.get_pos()
        pos = (min(max((pos[0]-env.joystick_offset[0]-20)/60, 0), 1),
               min(max((pos[1]-env.joystick_offset[1]-20)/60, 0), 1))
        obs, rew, done, info = env.step(np.array(pos, dtype=np.float32))
        print('Reward: '+str(rew))
        print('Score: '+str(info))
        t2 = time()
        dt = t2 - t1
        delay = (1/env.fps - dt)
        if delay < 0:
            print("[!] Can't keep framerate!")
        else:
            sleep(delay)


if __name__ == '__main__':
    main()
