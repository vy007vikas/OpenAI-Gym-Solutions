import gym
import numpy as np
import random
import tensorflow as tf

env = gym.make('CartPole-v0')
observation = env.reset()

EPISODES = 100
TIMESTAMP = 100
GAMMA = 0.9
ALPHA = 0.05
MOMENTUM = 0.9
explore_eps = 0.2


class neuralNet:
	def __init__(self,n,h,o):
		self.sess = tf.InteractiveSession()

		self.X = tf.placeholder(tf.float32,[None,n])
		self.Y = tf.placeholder(tf.float32,[None,1])
		self.W1 = tf.Variable(tf.zeros([n,h]))
		self.W2 = tf.Variable(tf.zeros([h,o]))
		self.B1 = tf.Variable(tf.zeros([h]))
		self.B2 = tf.Variable(tf.zeros([o]))

		self.sess.run(tf.initialize_all_variables())

		self.Y1 = tf.nn.relu(tf.matmul(self.X,self.W1) + self.B1)
		self.Y2 = tf.matmul(self.Y1,self.W2) + self.B2

		self.L = 0.5 * (self.Y - tf.reduce_max(self.Y2,reduction_indices=[1]))**2

		self.optimizer = tf.train.MomentumOptimizer(ALPHA, MOMENTUM)
		self.train_step = self.optimizer.minimize(self.L)

	def forward_pass(self,x):
		out = self.Y2.eval(feed_dict={self.X:x.reshape(1,x.shape[0])})
		return np.argmax(out) , np.max(out)

	def train(self,x,y):
		self.train_step.run(feed_dict={self.X:x , self.Y:y})


nnet = neuralNet(4,6,2)
data_batch = {}

def create_new_data(ot,re,ot2):
	data_batch['X'].append(ot)
	x , y = nnet.forward_pass(ot2)
	yval = re + GAMMA*y
	data_batch['Y'].append(yval)


for ep in range(EPISODES):
	observation = env.reset()
	reward = 0
	sum_reward = 0
	data_batch = {}
	data_batch['X'] = []
	data_batch['Y'] = []
	for t in range(TIMESTAMP):
		env.render()
		x = np.array(observation)
		action, actionval = nnet.forward_pass(x)
		print action, actionval

		tempvar = random.random()
		if tempvar < explore_eps:
			action = env.action_space.sample()

		observation, reward, done, info = env.step(action)
		create_new_data(x,reward,np.array(observation))

		new_batch = np.random.randint(1,data_batch['X'].shape[0], min(data_batch['X'].shape[0],50))
		nnet.train(data_batch['X'][new_batch,:] , data_batch['Y'][new_batch,:])

		sum_reward = sum_reward + reward
		if done :
			print("Episode finished after {} timesteps.", format(t+1))
			break


