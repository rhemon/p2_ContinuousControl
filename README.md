# Project 2: Continuous Control

### Introduction

For the second project in Udacity's Deep Reinforcement Learning, we had to train agent to work in the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, we were provided with two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Goal for the project

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  the agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Getting Started

1. I already have the environment for Windows 64 bit in the repository, but if you have some other operating system you can download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. If you download a new one then you will need to unzip the file here.
3. You will need to install unity agents for loading the environment. Note for this you will need Python 3.6.1 since with newer versions the required pytorch versions are not available.
    Once you have the right python version you can just run.
    ```
    pip install ./python
    ```
    This will install the required libraries


### Instructions

If you wish to train your own new agent, you can simply run the `train_single.py` file that makes use of the single agent version, but you can update the file in the line that reads the environment to the multip agent version as well.

I had to train the single agent one as the multi agent one was very slow in my laptop, since I ended up using all the GPU time provided in the workspace.

If you wish to keep track of the scores file like mine, you may wanna remove the `scores.npz` file, else it will add the new scores to my last track of scores.

If you also wish to just directly run the environment on the trained model, you can do so with the following code:
```
from unityagents import UnityEnvironment
from train_single import Agent
import torch
env = UnityEnvironment(file_name='Reacher/Reacher.exe') 
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name] # set False if you wish to visualize the environment
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]
agent = Agent(state_size, action_size, 0)
agent.network.actor.load_state_dict(torch.load('ddpg_actor_solution.pth'))
score = 0
for t in range(1000):
    action = agent.act(env_info.vector_observations)
    env_info = env.step(action)[brain_name]
    score += np.mean(env_info.rewards)
print("Final score:", score)
```

The above code will load the agent and iterate over it for 1000 step and take the action determiend by the actor model.

### Performance

Below are two clips, one shows the performance with untrained agent, the second shows the performance with trained agent.

![The Gif's meant to be here :/](./random.gif)
![The Gif's meant to be here :/](./trained_single.gif)

Although I trained it on version 1, I tried running it version 2 as well, and the performance was such:
![The Gif's meant to be here :/](./trained_mult.gif)

### Problems and Improvement

Above when you visualize the environment with trained agent you will notice it is not always able to go to the target location. So I think the model has high bias and could use more training. I did stop the training just after it achieved the average target score, so definitely training it for a little longer could have improved it a bit more.

I wanted to train the model using version 2 with multiple agents, I think it would give a much better policy. Especially it might be a good idea to implement D4PG or TRPO or TNPG as these models are likely to give a better performing agent. 
DDPG in my case was very unstable, and I had to do a lot of trial and error to come up with policy. That is why I was really happy the moment it achieved the targetted average score. 

Right now if I try the model on version 2, the average score over all agents appear lightly lower than the score of version 1 in most runs. Training with version 2 would have exposed the model to a lot more states, so a bit more of exploration. I also didn't really play with the values of noise, so maybe if I altered that and tried to make the model explore a little more than it could have been a little less underfitted.

I was also wondering it might be better if we used a common network for determining features that got fed into the actor and critic network. This might have made a good model for determining features and would have made it collect more useful information to take better actions.