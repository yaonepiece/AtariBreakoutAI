import numpy as np, tensorflow as tf
import random, time

class DQN:
	def __init__(self, input_size, output_size, max_memsize=5000, min_trainsize=500, trainstep=1, refresh_cycle=100, discount=0.99, epsilon=0.9, observe_max=3000000, learning_rate=0.001):
		#some variables
		self.actions=output_size
		self.steps=0
		self.max_memsize=max_memsize
		self.memsize=0
		self.min_trainsize=min_trainsize
		self.bagsize=min_trainsize
		self.trainstep=trainstep
		self.refresh_cycle=refresh_cycle
		self.trainflag=True
		self.times_trained=0
		self.discount=discount
		self.epsilon_init=epsilon
		self.epsilon=epsilon
		self.observe_max=observe_max
		
		#queries and targets
		self.xs=tf.placeholder(tf.float32, [None,input_size])
		self.ys=tf.placeholder(tf.float32, [None,output_size])
		
		#net1
		self.n1w1=tf.Variable(tf.random_normal([input_size,input_size]))
		self.n1b1=tf.Variable(tf.zeros([1,input_size])+0.1)
		self.n1c1=tf.matmul(self.xs,self.n1w1)+self.n1b1
		self.n1o1=tf.nn.relu(self.n1c1)
		self.n1w2=tf.Variable(tf.random_normal([input_size,output_size]))
		self.n1b2=tf.Variable(tf.zeros([1,output_size])+0.1)
		self.n1o2=tf.matmul(self.n1o1,self.n1w2)+self.n1b2
		self.n1loss=tf.reduce_mean(tf.reduce_sum(tf.square(self.ys-self.n1o2), reduction_indices=[1]))
		self.n1train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(self.n1loss)
		
		#net2
		self.n2w1=tf.Variable(tf.random_normal([input_size,input_size]))
		self.n2b1=tf.Variable(tf.zeros([1,input_size])+0.1)
		self.n2c1=tf.matmul(self.xs,self.n2w1)+self.n2b1
		self.n2o1=tf.nn.relu(self.n2c1)
		self.n2w2=tf.Variable(tf.random_normal([input_size,output_size]))
		self.n2b2=tf.Variable(tf.zeros([1,output_size])+0.1)
		self.n2o2=tf.matmul(self.n2o1,self.n2w2)+self.n2b2
		self.n2loss=tf.reduce_mean(tf.reduce_sum(tf.square(self.ys-self.n2o2), reduction_indices=[1]))
		self.n2train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(self.n2loss)
		
		#session
		self.sess=tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.sess.graph.finalize()
		
		#memory unit
		self.laststate=None
		self.smem=np.empty([0,input_size],dtype=float)
		self.rmem=np.empty([0],dtype=float)
		self.mmem=np.empty([0],dtype=int)
		self.s_mem=np.empty([0,input_size],dtype=float)
	
	def action(self,state):
		if np.random.rand()<self.epsilon:
			values=self.sess.run(self.n1o2, feed_dict={self.xs: [state]}).ravel()
			for i in range(self.actions):
				if values[i]==np.amax(values):
					return i
		else:
			return np.random.randint(self.actions)
	
	def train(self):
		sampleind=random.sample(range(self.memsize),k=self.bagsize)
		x=self.smem[sampleind,:]
		x_=self.s_mem[sampleind,:]
		r=self.rmem[sampleind]
		m=self.mmem[sampleind]
		if self.trainflag:
			y=self.sess.run(self.n1o2,feed_dict={self.xs:x})
			y_=self.sess.run(self.n1o2,feed_dict={self.xs:x_})
			y_tar=self.sess.run(self.n2o2,feed_dict={self.xs:x_})
			for i in range(len(y)):
				am=0
				a=y_[i]
				for j in range(self.actions):
					if a[j]==np.amax(a):
						am=j
						break
				y[i][m[i]]=r[i]+y_tar[i][am]*self.discount
			self.sess.run(self.n1train_step,feed_dict={self.xs:x,self.ys:y})
		else:
			y=self.sess.run(self.n2o2,feed_dict={self.xs:x})
			y_=self.sess.run(self.n2o2,feed_dict={self.xs:x_})
			y_tar=self.sess.run(self.n1o2,feed_dict={self.xs:x_})
			for i in range(len(y)):
				am=0
				a=y_[i]
				for j in range(self.actions):
					if a[j]==np.amax(a):
						am=j
						break
				y[i][m[i]]=r[i]+y_tar[i][am]*self.discount
			self.sess.run(self.n2train_step,feed_dict={self.xs:x,self.ys:y})
	
	def dec_mem(self):
		maxsize=max(self.min_trainsize*2,self.max_memsize*0.6)
		self.memsize=np.random.randint(self.min_trainsize*2,maxsize)
		self.smem=self.smem[-self.memsize:]
		self.rmem=self.rmem[-self.memsize:]
		self.mmem=self.mmem[-self.memsize:]
		self.s_mem=self.s_mem[-self.memsize:]
	
	def play(self,env):
		if self.laststate is None:
			self.laststate=env.getState()
		
		#make a move
		m=self.action(self.laststate)
		r1,r2,r3,r4=0,0,0,0
		_, r1, ter=env.render(m)
		if not ter:
			_, r2, ter=env.render(m)
		if not ter:
			_, r3, ter=env.render(m)
		if not ter:
			_, r4, ter=env.render(m)
		r=r1+r2+r3+r4
		s_=env.getState()
		
		#update memories
		#if self.steps<self.observe_max:
		if True:
			self.smem=np.append(self.smem,[self.laststate],axis=0)
			self.rmem=np.append(self.rmem,[r],axis=0)
			self.mmem=np.append(self.mmem,[m],axis=0)
			self.s_mem=np.append(self.s_mem,[s_],axis=0)
			self.memsize+=1
			if self.memsize>self.max_memsize:
				self.dec_mem()
			if ter:
				env.restart()
				self.laststate=None
			else:
				self.laststate=np.copy(s_)
			self.bagsize=max(self.min_trainsize,int(self.memsize*0.4))
		
		#train
		#if self.steps<self.observe_max and self.memsize>self.min_trainsize:
		if self.memsize>self.min_trainsize:
			self.steps+=1
			if self.steps<self.observe_max:
				self.epsilon=self.epsilon_init+(1-self.epsilon_init)*(self.steps/self.observe_max)
			else:
				self.epsilon=1
			if self.steps%self.trainstep==0:
				self.train()
				self.times_trained+=1
				if self.times_trained%self.refresh_cycle==0:
					self.trainflag=not self.trainflag
		
		#log
		self.log(r)
		
	def log(self,r):
		if self.steps==0:
			state='Collect'
		elif self.steps<=self.observe_max:
			state='Observe'
		else:
			state='Play'
		print('Steps:{0} State:{1} Epsilon:{2:.3f} Reward:{3}'.format(self.steps,state,self.epsilon,r))

class dumbGame:
	def __init__(self):
		self.state=np.array([0,0],dtype=int)
		self.record=np.zeros([4],dtype=int)
	
	def getState(self):
		return self.state
	
	def render(self,move):
		reward=0
		if move>0:
			self.state[move-1]=1-self.state[move-1]
		if np.sum(self.state)==2:
			reward=1
		if np.sum(self.state)==0:
			reward=-1
		self.record[self.state[0]+self.state[1]*2]+=1
		return 0, reward, False

if __name__=='__main__':
	game=dumbGame()
	bot=DQN(2,3,30,5,1,5,0.499999,0.7,1000,0.1)
	for i in range(1005):
		bot.play(game)
		if bot.steps%10==0 and bot.steps>0:
			print('Steps:{0} Record:{1}'.format(bot.steps,game.record))
			game.record=np.zeros([4],dtype=int)
	print('test successful')