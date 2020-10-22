import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Deep Q Learning
# Training images:
#   -> CartPole-v0: images/cartpole-dqn.png

# Constants
SEED = 1
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4               # State space size
ACT_N = 2               # Action space size
EPSILON = 1.0           # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1       # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25       # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 300          # Total number of episodes to learn over
TEST_EPISODES = 10      # Test episodes

# Create environment
utils.seed.seed(SEED)
env = gym.make("CartPole-v0")
env.seed(SEED)

# Infinite play with the environment (uncomment)
## policy = lambda env, obs: np.random.choice(env.action_space.n)
## while True: utils.envs.play_episode(env, policy, render = True)

# Create replay buffer
buf = utils.buffers.ReplayBuffer(BUFSIZE)

# Create network for Q(s, a)
q = torch.nn.Sequential(
    torch.nn.Linear(OBS_N, 100), torch.nn.ReLU(),
    torch.nn.Linear(100, 50), torch.nn.ReLU(),
    torch.nn.Linear(50, ACT_N)
).to(DEVICE)

# Create epsilon-greedy policy
def policy(env, obs):

    global EPSILON, STEPS, STEPS_MAX
    obs = t.f(obs).view(-1, OBS_N)  # Convert to torch tensor
    
    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        qvalues = q(obs)
        action = torch.argmax(qvalues).item()
    
    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action


# Create training function
OPT = torch.optim.Adam(q.parameters(), lr = LEARNING_RATE)
LOSSFN = torch.nn.MSELoss()
def train(q, buf):

    global OPT, LOSSFN
    
    # Sample a minibatch (s, a, r, s', d)
    # Each variable is a vector of corresponding values
    S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)
    
    # Get Q(s, a) for every (s, a) in the minibatch
    qvalues = q(S).gather(1, A.view(-1, 1)).squeeze()

    # Get max_a' Q(s', a') for every (s') in the minibatch
    q2values = torch.max(q(S2), dim = 1).values

    # If done, 
    #   y = r(s, a) + GAMMA * max_a' Q(s', a') * (0)
    # If not done,
    #   y = r(s, a) + GAMMA * max_a' Q(s', a') * (1)       
    targets = R + GAMMA * q2values * (1-D)

    # Detach y since it is the target. Target values should
    # be kept fixed.
    loss = LOSSFN(targets.detach(), qvalues)

    # Backpropagation
    OPT.zero_grad()
    loss.backward()
    OPT.step()

    return loss.item()

# Play episodes
Rs = [] 
last25Rs = []
print("Training:")
pbar = tqdm.trange(EPISODES)
for epi in pbar:

    # Play an episode and log episodic reward
    S, A, R = utils.envs.play_episode_rb(env, policy, buf)
    Rs += [sum(R)]

    # Train after collecting sufficient experience
    if epi >= TRAIN_AFTER_EPISODES:

        # Train for TRAIN_EPOCHS
        # Loss is returned, can be plotted (TODO)
        for tri in range(TRAIN_EPOCHS): 
            train(q, buf)

    # Show mean episodic reward over last 25 episodes
    last25Rs += [sum(Rs[-25:])/len(Rs[-25:])]
    pbar.set_description("R25(%g)" % (last25Rs[-1]))

pbar.close()
print("Training finished!")

# Plot the reward
N = len(last25Rs)
plt.plot(range(N), last25Rs, 'b')
plt.savefig("images/cartpole-dqn.png")
print("Episodic reward plot saved!")

# Play test episodes
print("Testing:")
for epi in range(TEST_EPISODES):
    S, A, R = utils.envs.play_episode(env, policy, render = True)
    print("Episode%02d: R = %g" % (epi+1, sum(R)))
env.close()