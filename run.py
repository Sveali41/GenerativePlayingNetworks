from models.generator import Generator
from agents.ppoagent import PPOAgent
from trainer import Trainer
from game.env import Env


def main(game_name, game_length):
	#Game description
	env = Env(game_name, game_length)

	#Network
	latent_shape = (100,)
	gen = Generator(latent_shape, env)

        #Here: verify agents reward going the right way
	#Run without generator to see training
        #Focus on doing better than random initialization (maybe don't update generator)

	#Agent
	num_processes = 24
	experiment = "experiment_gen"
	agent = PPOAgent(env, num_processes, experiment) #, lr=.001)

	#Training
	t = Trainer(gen, agent, experiment, 1)
	t.train(1000, 32, 1e6) #1000, 32, 8192
	#t.train(10e6, 8192, 32) #10m training steps, in batches of 8192 steps per 32 levels

if(__name__ == "__main__"):
	main('zelda', 1000)
