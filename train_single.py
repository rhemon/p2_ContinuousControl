from unityagents import UnityEnvironment
import numpy as np
from collections import deque, namedtuple

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque
import matplotlib.pyplot as plt


BUFFER_SIZE = int(1e6)    # Memory size, the last number of steps it stores
BATCH_SIZE = 64           # Batch size used to train the model 
ACTOR_LR = 1e-4           # Learning rate for the Actor model
CRITIC_LR = 1e-4          # Learning rate for the Critic model
WEIGHT_DECAY = 0          # Weight decay (used for Critic)
TAU = 1e-3                # TAU used to determine how much of local impacts the target
GAMMA = 0.99              # Discount rate
UPDATE_EVERY = 20         # Learn every few steps 
TIMES_UPDATE = 10         # Number of time to learn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    """
    Returns lower and upper limit for the random parameterial initalisation for given
    layer.

    :param layer: torch's neural network layer
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class CriticNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed):
        """
        Initialize critic network.

        :param state_size: Number of information provided in the state
        :param action_size: Number of actions environment can take
        :param seed: Seed for random initialization
        """
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.critic_layer_1 = nn.Linear(state_size, 256)
        self.critic_layer_2 = nn.Linear(256+action_size, 128)
        self.critic_out = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters initialized for the layers in the network.
        """
        self.critic_layer_1.weight.data.uniform_(*hidden_init(self.critic_layer_1))
        self.critic_layer_2.weight.data.uniform_(*hidden_init(self.critic_layer_2))
        self.critic_out.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        """
        Feedforward the the network with provided state and action. Returns the Q value.
        
        :param state: The state of the game
        :param action: Action taken by the agent
        """
        x = F.leaky_relu(self.critic_layer_1(state))
        x = torch.cat([x, action], dim=1)
        x = F.leaky_relu(self.critic_layer_2(x))
        return self.critic_out(x)

class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        """
        Initialize critic network.

        :param state_size: Number of information provided in the state
        :param action_size: Number of actions environment can take
        :param seed: Seed for random initialization
        """
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.actor_layer_1 = nn.Linear(state_size, 256)
        self.actor_layer_2 = nn.Linear(256, 128)
        self.actor_out = nn.Linear(128, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters initialized for the layers in the network.
        """
        self.actor_layer_1.weight.data.uniform_(*hidden_init(self.actor_layer_1))
        self.actor_layer_2.weight.data.uniform_(*hidden_init(self.actor_layer_2))
        self.actor_out.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        """
        Feedforward the the network with provided state. Returns the action values.
        
        :param state: The state of the game
        """
        x = F.leaky_relu(self.actor_layer_1(state))
        x = F.leaky_relu(self.actor_layer_2(x))
        return torch.tanh(self.actor_out(x))

class Network:
    def __init__(self, state_size, action_size, seed):
        """
        Initializes and handles both actor and critic network for the agent.

        :param state_size: Number of information provided in the state
        :param action_size: Number of actions environment can take
        :param seed: Seed for random initialization
        """
        self.seed = seed
        self.actor = ActorNetwork(state_size, action_size, seed).to(device)
        self.critic = CriticNetwork(state_size, action_size, seed).to(device)

    def copy(self, network):
        """
        Copy the parameters from the provided network into current actor critic networks.

        :param network: Network instance with actor and critic model initialized
        """
        # Actor
        for target_param, local_param in zip(self.actor.parameters(), network.actor.parameters()):
            target_param.data.copy_(local_param.data)
        
        # Critic
        for target_param, local_param in zip(self.critic.parameters(), network.critic.parameters()):
            target_param.data.copy_(local_param.data)
    
    def soft_update(self, network, tau=TAU):
        """
        Soft update the current network from the given network.

        :param network: Network values that will be used to update current network
        :param tau: Floating value to determine how much information goes into current network from provided one.
        """
        # Actor
        for target_param, local_param in zip(self.actor.parameters(), network.actor.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        # Critic
        for target_param, local_param in zip(self.critic.parameters(), network.critic.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Agent:

    def __init__(self, state_size, action_size, random_seed, action_low=-1, action_high=1):
        """
        Initializes the agent to play in the environment.
        
        :param state_size: Number of information provided in the state
        :param action_size: Number of actions environment can take
        :param random_seed: Seed for random initialization
        :param action_low: Minimum value for action
        :param action_high: Maxmimum value for aciton
        """
        
        self.seed = random.seed(random_seed)
        
        self.state_size = state_size
        self.action_size = action_size
        self.a_low = action_low
        self.a_high = action_high
        self.network = Network(state_size, action_size, random_seed)
        
        self.actor_opt = optim.Adam(self.network.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = optim.Adam(self.network.critic.parameters(), lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)

        self.target_network = Network(state_size, action_size, random_seed)
        
        self.ounoise = OUNoise(action_size, action_low, action_high)

        self.memory = ReplayBuffer()
        self.t_step = 0
    
    def act(self, state, add_noise=True):
        """
        Returns action for given state.

        :param state: State of the environment,for which to determine an action
        :param add_noise: Used to determine whether to add nose based on Ornstein Uhlenbeck process
        """
        state = torch.tensor(state).float().to(device)
        self.network.actor.eval()
        with torch.no_grad():
            action = self.network.actor(state)
            action = action.data.cpu().numpy()
        self.network.actor.train()
        if add_noise:
            return self.ounoise.get_action(action)
        return action
    
    def step(self, state, action, reward, next_state, done):
        """
        Add the step to memory and if its time to update the model, then train and update the networks.

        :param state: State of the environment
        :param action: Action taken for that state
        :param reward: Reward provided for the taken action
        :param next_state: New state that appeared due to action
        :param done: Whehter its the terminal state or not
        """
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for i in range(TIMES_UPDATE):
                experiences = self.memory.sample()
                self.learn(experiences)
                self.target_network.soft_update(self.network)
    
    def learn(self, experiences, gamma=GAMMA):
        """
        Learning algorithm for the model. Uses the target critic network to determine
        the MSE loss for predicted Q values with local network for the experieces sampled 
        from the memory. In the backpropagation of critic network, clips the gradient values to 1.
        Then updates the actor network with the goal to maximize the average value determined by the critic model.
        So the loss is -Q_local(state, action).mean(). 

        :param experiences: State, Actions, Rewards, Next states, dones randomly sampled from the memory
        :param gamma: Discount rate that determines how much of future reward impacts total reward.
        """
        states, actions, rewards, next_states, dones = experiences

        Q_target_next = self.target_network.critic(next_states, self.target_network.actor(next_states))
        Q_target = rewards + (gamma * Q_target_next * (1-dones))
        
        Q_predicted = self.network.critic(states, actions)
        critic_loss = F.mse_loss(Q_predicted, Q_target)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.critic.parameters(), 1)
        self.critic_opt.step()

        actor_loss = -self.network.critic(states, self.network.actor(states)).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""
class OUNoise(object):
    def __init__(self, action_size, a_low, a_high, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        """
        Initialize the parameters to process the noise values that will be added
        to the actions.
        """
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_size
        self.low          = a_low
        self.high         = a_high
        self.reset()
        
    def reset(self):
        """
        Reset current state of the process
        """
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        """
        Update the state
        """
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        """
        For given action add the noise to the action and return it clipped to its range.
        """
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

def ddpg(agent, env, brain_name, target_score=30, max_t=1000, gamma=GAMMA):
    """
    Training function used to expose the agent to the environment and run through it repeated until
    target score is achieved.

    :param agent: Agent that will be trained
    :param env: Environment the agent will be learning
    :param brain_name: Brain name of the environment
    :param target_score: Target score the agent needs to achieve
    :param max_t: Number of times to iterate each episode
    :param gammma: Discount rate
    """
    try:
        scores = list(np.load('scores.npz')['scores'])
    except:
        scores = []
    
    scores_window = deque(maxlen=100)
    i_ep = 0
    while True:
        i_ep += 1
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            for i in range(num_agents):
                state = states[i]
                action = actions[i]
                reward = rewards[i]
                next_state = next_states[i]
                done = dones[i]
                agent.step(state, action, reward, next_state, done)
            score += rewards
            states = next_states
            if done:
                break
                
        scores_window.append(score.mean())
        scores.append(score.mean())
        print('\rEpisode {} \tAverage Score: {:.2f}, Max score: {}, Min score: {}'.format(i_ep, np.mean(scores_window), np.max(scores_window), np.min(scores_window)), end="")
        np.savez("scores.npz", scores=scores)
        if i_ep % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_ep, np.mean(scores_window)))
            torch.save(agent.network.actor.state_dict(), f'ddpg_actor_checkpoint.pth') 
            torch.save(agent.network.critic.state_dict(), f'ddpg_critic_checkpoint.pth')

        if np.mean(scores_window) > target_score and i_ep > 100:
            print('\rSolved goal on episode {} with average score {}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.network.actor.state_dict(), f'ddpg_actor_solution.pth') 
            torch.save(agent.network.critic.state_dict(), f'ddpg_critic_solution.pth')
            break
        
    return scores

if __name__ == "__main__":
    env = UnityEnvironment(file_name='Reacher/Reacher.exe')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    agent = Agent(state_size, action_size, 1)

    # agent.network.actor.load_state_dict(torch.load('ddpg_actor_checkpoint.pth'))
    # agent.network.critic.load_state_dict(torch.load('ddpg_critic_checkpoint.pth'))

    scores = ddpg(agent, env, brain_name, max_t=1000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()