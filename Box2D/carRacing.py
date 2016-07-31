import gym
import numpy as np
import random
from scipy import misc
import tensorflow as tf

env = gym.make('CarRacing-v0')
observation = env.reset()

EPISODES = 0
TIMESTAMP = 1
GAMMA = 0.99
ALPHA = 0.001
explore_eps = 1
N = 50
OUT = 3
BATCH_SIZE = 4

def conv2d(x,W,stride):
	return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding='SAME')

class neuralNet:
	def __init__(self):
		self.sess = tf.InteractiveSession()

		self.X = tf.placeholder(tf.float32,[None,N,N,1])
		self.C = tf.placeholder(tf.float32,[None,OUT])
		self.Y = tf.placeholder(tf.float32,[None,OUT])

		self.W_conv1 = tf.Variable(tf.truncated_normal([8,8,1,32],stddev = 0.1))        # 50 * 50 * 1
		self.B_conv1 = tf.Variable(tf.zeros([32]))

		self.W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev = 0.1))       # 15 * 15 * 32
		self.B_conv2 = tf.Variable(tf.zeros([64]))

		self.W_conv3 = tf.Variable(tf.truncated_normal([3,3,64,64],stddev = 0.1))       # 6 * 6 * 64
		self.B_conv3 = tf.Variable(tf.zeros([64]))

		self.W_fc1 = tf.Variable(tf.truncated_normal([ 4*4*64 , 512],stddev = 0.1))     # 4 * 4 * 64
		self.B_fc1 = tf.Variable(tf.zeros([512]))

		self.W_fc2 = tf.Variable(tf.truncated_normal([512,OUT],stddev = 0.1))
		self.B_fc2 = tf.Variable(tf.zeros([OUT]))

		o_conv1 = tf.nn.relu(conv2d(self.X,self.W_conv1,3) + self.B_conv1)
		o_pool1 = max_pool_2x2(o_conv1)

		o_conv2 = tf.nn.relu(conv2d(o_pool1,self.W_conv2,2) + self.B_conv2)

		o_conv3 = tf.nn.relu(conv2d(o_conv2,self.W_conv3,1) + self.B_conv3)
		o_fconv3 = tf.reshape(o_conv3,[-1,4*4*64])

		o_fc1 = tf.nn.relu(tf.matmul(o_fconv3,self.W_fc1) + self.B_fc1)

		self.o_fc2 = tf.matmul(o_fc1,self.W_fc2) + self.B_fc2

		self.L = tf.reduce_sum(tf.square(self.Y - tf.mul(self.o_fc2,self.C)))

		self.optimizer = tf.train.AdamOptimizer(ALPHA)
		self.train_step = self.optimizer.minimize(self.L)

		self.sess.run(tf.initialize_all_variables())

	def forward_pass(self,x):
		out = self.o_fc2.eval(feed_dict={self.X:x.reshape(1,x.shape[0])})
		# print out
		return np.argmax(out) , np.max(out)

	def train(self,x,y,c):
		self.train_step.run(feed_dict={self.X:x , self.Y:y, self.C:c})


def sanity_check():
	observation = env.reset()
	print observation.shape
	print(env.action_space)
	print(env.observation_space)
	print(env.observation_space.high)
	print(env.observation_space.low)

def process_image(ot):
	ot = misc.imresize(ot , (50,50,3) )
	ot = 0.299*ot[:,:,0] + 0.587*ot[:,:,1] + 0.114*ot[:,:,2]
	ot = np.reshape(ot , (N,N,1))
	return ot

def create_new_data(ot,re,ot2,reset,done,action):
	c = np.zeros((1,OUT))
	c[0][action] = 1
	yval = np.zeros((1,OUT))
	x , y = nnet.forward_pass(ot2)
	yval[0][action] = re
	if not done:
		yval[0][action] = re + GAMMA*y
	data_batch['C'] = c
	if reset:
		data_batch['X'] = ot
		data_batch['Y'] = yval
	else:
		data_batch['X'] = np.append(data_batch['X'],ot,axis=0)
		data_batch['Y'] = np.append(data_batch['Y'],yval,axis=0)

nnet = neuralNet()
data_batch = {}
ans = 0
ans1 = 0
for ep in range(EPISODES):
	observation = env.reset()
	observation = process_image(observation)
	reward = 0
	sum_reward = 0
	data_batch = {}
	reset = True
	for t in range(TIMESTAMP):
		env.render()
		x = np.array(observation)
		action, actionval = nnet.forward_pass(x)
		# print action, actionval

		tempvar = random.random()
		if tempvar < max((500/(ep+1)),explore_eps) and ep < 9000:      # dont explore for last 1000 episodes
			action = env.action_space.sample()

		# print action
		observation, reward, done, info = env.step(action)
		observation = process_image(observation)
		create_new_data(x,reward,np.array(observation),reset,done,action)
		reset = False

		if data_batch['X'].shape[0] == BATCH_SIZE:
			nnet.train(data_batch['X'] , data_batch['Y'], data_batch['C'])
			reset = True

		sum_reward = sum_reward + reward
		if done :
			nnet.train(data_batch['X'] , data_batch['Y'], data_batch['C'])
			print("Episode finished after {} timesteps.", format(t+1))
			ans = max(ans,t)
			if ep > 9000:
				ans1 = max(ans1,t)
			break

print ans , ans1
