# Project Columbus

Project Columbus is a framework for trivial 2D OpenAI Gym environments that are supposed to test a agents ability to solve tasks that require different forms of exploration effectively and efficiently.  

![Screenshot](./img_README.png)

### env.py
Contains the ColumbusEnv. New envs are implemented by subclassing ColumbusEnv and expanding _init_ and overriding _setup_.

### entities.py
Contains all implemented entities (e.g. the Agent, Rewards and Enemies)

### observables.py
Contains all 'oberservables'. These are attached to envs to define what kind of output is given to the agent. This way environments can be designed independently from the observation machanism that is used by the agent to play it.

### humanPlayer.py
Allows environments to be played by a human using mouse input.
