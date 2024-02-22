from models.generator import Generator
from agents.agent import Agent
from trainer import Trainer
from game.env import Env
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from paths import *
from stable_baselines3.common.callbacks import BaseCallback


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.factor = 0.99
        self.cumulate_reward = []
        self.cumulate_reward_factor = []

    def _on_step(self) -> bool:
        # Access and print the reward
        # self.locals contains the rollout data
        reward = self.locals['rewards'][-1]  # Assuming a single environment
        self.cumulate_reward.append(reward)
        t = len(self.cumulate_reward)
        if t == 1:
            G = reward
        else:
            G = (self.factor ** t) * reward + self.cumulate_reward_factor[-1]
        self.cumulate_reward_factor.append(G)
        # print(f"Reward: {reward}")
        return True


def gen_env(self, updates, batch_size, gen_batches, div_batches, rl_steps, pretrain):
    self.generator.train()
    z = self.z_generator(batch_size, self.generator.z_size)  # 128 scale debug
    lvl_strs, states = self.new_elite_levels(z(batch_size))  # gen_batch 32
    # visualize the levels
    for i in range(len(lvl_strs)):
        # the batch of gen is 32 -> 32 levels for agents
        pic = self.level_visualizer.draw_level(lvl_strs[i])
        pic.show()


def set_env(trainer, num_envs=None):
    path = Paths
    with open(path.LEVEL_FILE, 'r') as file:
        # Read the entire file content into a string
        lvl_strs = file.read()
        # the batch of gen is 32 -> 32 levels for agents
    pic = trainer.level_visualizer.draw_level(lvl_strs)
    pic.show()

# Game description
reward_mode = 'base'
reward_scale = 1.0
elite_prob = 0
env = Env('zelda', 1000,
          {'reward_mode': reward_mode, 'reward_scale': reward_scale, 'elite_prob': elite_prob})

# Network
latent_shape = (512,)
dropout = 0
lr = .0001
gen = Generator(latent_shape, env, 'nearest', dropout, lr)

# Agent
num_processes = 1
experiment = "Experiments"
lr = .00025
model = 'base'
dropout = .3
reconstruct = None
r_weight = .05
Agent.num_steps = 5
Agent.entropy_coef = .01
Agent.value_loss_coef = .1
agent = Agent(env, num_processes, experiment, 0, lr, model, dropout, reconstruct, r_weight)

# Training
gen_updates = 200  # 1e4
gen_batch = 1  # 32
gen_batches = 1
diversity_batches = 0
rl_batch = 100  # agent is trained on these levels for a specified number of steps   1e4
pretrain = 0
elite_persist = False
elite_mode = 'mean'
load_version = 0
notes = ''
agent.writer.add_hparams(
    {'Experiment': experiment, 'RL_LR': lr, 'Minibatch': gen_batch, 'RL_Steps': rl_batch, 'Notes': notes}, {})
t = Trainer(gen, agent, experiment, load_version, elite_mode, elite_persist)
t.loss = lambda x, y: x.mean().pow(2)
set_env(t)  # visualize
# env test
# for i in range(10):
#     done = False
#     obs = t.agent.envs.reset()
#     while not done:
#         random_action = t.agent.envs.action_space.sample()
#         print("action", random_action)
#         obs, reward, done, info = t.agent.envs.step(random_action)
#         print("reward", reward)

model = A2C('MlpPolicy', t.agent.envs, verbose=1)
# Initialize the custom callback
callback = RewardCallback()
# Train the model with the custom callback
model.learn(total_timesteps=10000, callback=callback)

# Plotting the discounted cumulative rewards
plt.plot(callback.cumulate_reward_factor)
plt.xlabel('Episode')
plt.ylabel('Discounted Cumulative Reward')
plt.title('Discounted Cumulative Reward per Episode')
plt.show()

# test the trained agent
obs = t.agent.envs.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print("Step{}".format(step + 1))
    print("Action:", action)
    obs, reward, done, info = t.agent.envs.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    if done:
        print("Goal reached!", "reward=", reward)
        break
