import gym
import numpy as np
import random
from scipy import misc
import tensorflow as tf

env = gym.make('CarRacing-v0')
observation = env.reset()

EPISODES = 1
TIMESTAMP = 5
GAMMA = 0.99
ALPHA = 0.001
explore_eps = 1
N = 50
OUT1 = 5
OUT2 = 5
OUT3 = 5
BATCH_SIZE = 4

def conv2d(x,W,stride):
	return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding='SAME')

class neuralNet:
	def __init__(self):
		self.sess = tf.InteractiveSession()

		self.X = tf.placeholder(tf.float32,[None,N,N,1])
		self.C1 = tf.placeholder(tf.float32,[None,OUT1])
		self.C2 = tf.placeholder(tf.float32,[None,OUT2])
		self.C3 = tf.placeholder(tf.float32,[None,OUT3])
		self.Y1 = tf.placeholder(tf.float32,[None,OUT1])
		self.Y2 = tf.placeholder(tf.float32,[None,OUT2])
		self.Y3 = tf.placeholder(tf.float32,[None,OUT3])

		self.W_conv1 = tf.Variable(tf.truncated_normal([8,8,1,32],stddev = 0.1))        # 50 * 50 * 1
		self.B_conv1 = tf.Variable(tf.zeros([32]))

		self.W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev = 0.1))       # 15 * 15 * 32
		self.B_conv2 = tf.Variable(tf.zeros([64]))

		self.W_conv3 = tf.Variable(tf.truncated_normal([3,3,64,64],stddev = 0.1))       # 7 * 7 * 64
		self.B_conv3 = tf.Variable(tf.zeros([64]))

		self.W_fc1 = tf.Variable(tf.truncated_normal([ 5*5*64 , 512],stddev = 0.1))     # 5 * 5 * 64
		self.B_fc1 = tf.Variable(tf.zeros([512]))

		self.W_fc21 = tf.Variable(tf.truncated_normal([512,OUT1],stddev = 0.1))
		self.B_fc21 = tf.Variable(tf.zeros([OUT1]))

		self.W_fc22 = tf.Variable(tf.truncated_normal([512,OUT2],stddev = 0.1))
		self.B_fc22 = tf.Variable(tf.zeros([OUT2]))

		self.W_fc23 = tf.Variable(tf.truncated_normal([512,OUT3],stddev = 0.1))
		self.B_fc23 = tf.Variable(tf.zeros([OUT3]))

		o_conv1 = tf.nn.relu(conv2d(self.X,self.W_conv1,3) + self.B_conv1)
		o_pool1 = max_pool_2x2(o_conv1)

		o_conv2 = tf.nn.relu(conv2d(o_pool1,self.W_conv2,2) + self.B_conv2)

		o_conv3 = tf.nn.relu(conv2d(o_conv2,self.W_conv3,1) + self.B_conv3)
		o_fconv3 = tf.reshape(o_conv3,[-1,5*5*64])

		o_fc1 = tf.nn.relu(tf.matmul(o_fconv3,self.W_fc1) + self.B_fc1)

		self.o_fc21 = tf.matmul(o_fc1,self.W_fc21) + self.B_fc21

		self.o_fc22 = tf.matmul(o_fc1,self.W_fc22) + self.B_fc22

		self.o_fc23 = tf.matmul(o_fc1,self.W_fc23) + self.B_fc23

		self.L1 = tf.reduce_sum(tf.square(self.Y1 - tf.mul(self.o_fc21,self.C1)))
		self.L2 = tf.reduce_sum(tf.square(self.Y2 - tf.mul(self.o_fc22,self.C2)))
		self.L3 = tf.reduce_sum(tf.square(self.Y3 - tf.mul(self.o_fc23,self.C3)))

		self.optimizer = tf.train.AdamOptimizer(ALPHA)
		self.train_step1 = self.optimizer.minimize(self.L1)
		self.train_step2 = self.optimizer.minimize(self.L2)
		self.train_step3 = self.optimizer.minimize(self.L3)

		self.sess.run(tf.initialize_all_variables())

	def forward_pass(self,x):
		with self.sess.as_default():
			out1, out2, out3 = self.sess.run([self.o_fc21,self.o_fc22,self.o_fc23],feed_dict={self.X:x})
		# print out
		return np.argmax(out1),np.argmax(out2),np.argmax(out3),np.max(out1),np.max(out2),np.max(out3)

	def train(self,x,y1,y2,y3,c1,c2,c3):
		with self.sess.as_default():
			self.sess.run([self.train_step1,self.train_step2,self.train_step3],feed_dict={
				self.X:x , self.Y1:y1, self.Y2:y2, self.Y3:y3, self.C1:c1, self.C2:c2, self.C3:c3})


def sanity_check():
	observation = env.reset()
	print observation.shape
	print(env.action_space)
	print(env.action_space.sample())
	print(env.observation_space)
	# print(env.observation_space.high)
	# print(env.observation_space.low)
	print(env.action_space.high)
	print(env.action_space.low)

def process_image(ot):
	ot = misc.imresize(ot , (N,N,3) )
	ot = 0.299*ot[:,:,0] + 0.587*ot[:,:,1] + 0.114*ot[:,:,2]
	ot = np.reshape(ot , (1,N,N,1))
	return ot

def create_new_data(ot,re,ot2,reset,done,a1,a2,a3):
	c1 = np.zeros((1,OUT1))
	c1[0][a1] = 1
	c2 = np.zeros((1,OUT1))
	c2[0][a2] = 1
	c3 = np.zeros((1,OUT1))
	c3[0][a3] = 1
	yval1 = np.zeros((1,OUT1))
	yval2 = np.zeros((1,OUT2))
	yval3 = np.zeros((1,OUT3))
	b1,b2,b3,bv1,bv2,bv3 = nnet.forward_pass(ot2)
	yval1[0][a1] = re
	yval2[0][a2] = re
	yval3[0][a3] = re
	if not done:
		yval1[0][a1] = re + GAMMA*bv1
		yval2[0][a2] = re + GAMMA*bv2
		yval3[0][a3] = re + GAMMA*bv3
	data_batch['C1'] = c1
	data_batch['C2'] = c2
	data_batch['C3'] = c3
	if reset:
		data_batch['X'] = ot
		data_batch['Y1'] = yval1
		data_batch['Y2'] = yval1
		data_batch['Y3'] = yval1
	else:
		data_batch['X'] = np.append(data_batch['X'],ot,axis=0)
		data_batch['Y1'] = np.append(data_batch['Y1'],yval1,axis=0)
		data_batch['Y2'] = np.append(data_batch['Y2'],yval2,axis=0)
		data_batch['Y3'] = np.append(data_batch['Y3'],yval3,axis=0)

nnet = neuralNet()
data_batch = {}
sanity_check()
ans = np.zeros((12))
anssum = np.zeros((12))
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
		a1,a2,a3,av1,av2,av3 = nnet.forward_pass(x)

		tempvar = random.random()
		if tempvar < max((500/(ep+1)),explore_eps) and ep < 9000:      # dont explore for last 1000 episodes
			a1 = np.random.randint(0,5,size=1)
			a2 = np.random.randint(0,5,size=1)
			a3 = np.random.randint(0,5,size=1)

		action = [ -1.0 + a1*0.4 , a2*0.2 , a3*0.2 ]
		observation, reward, done, info = env.step(action)
		observation = process_image(observation)
		create_new_data(x,reward,np.array(observation),reset,done,a1,a2,a3)
		print data_batch['X'].shape , data_batch['Y'].shape , data_batch['C'].shape
		reset = False

		if data_batch['X'].shape[0] == BATCH_SIZE:
			nnet.train(data_batch['X'] , data_batch['Y'], data_batch['C'])
			reset = True

		sum_reward = sum_reward + reward
		if done or t == TIMESTAMP-1:
			nnet.train(data_batch['X'] , data_batch['Y'], data_batch['C'])
			print("Episode {0} finished after {1} timesteps.".format(ep+1,t+1))
			ans[int(ep/5000)] = max(ans[int(ep/5000)],t)
			anssum[int(ep/5000)] += anssum[int(ep/5000)]
			break

for i in range(3):
	print (i*5000 , " -- ", (i+1)*5000 , " == " , ans[i] , (anssum[i]/5000))
