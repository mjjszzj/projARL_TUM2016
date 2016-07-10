#!/usr/bin/env python

import array
import serial
import struct
import binascii
import numpy as np
from geometry_msgs.msg import Pose2D, Accel, PoseArray, Vector3, Quaternion
import tf
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import rosbag
import math
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


class ENV(object):
    state_dim = 3
    action_num = 2
    state = None
    a_num = None
    q_w = None
    acc = 0.1
    freq = 10.
    target = 1
    room_size = 3
    speedlimit = 0.5
    
    def __init__(self):
        self.state = np.zeros(self.state_dim)
        self.a_num = self.action_num
        
    def reset(self):
        self.state = np.zeros(self.state_dim)
        return self.state
        
    def step(self, action, start=False):
        reward = 0
        if start:
            return self.state, reward, False
        ''' state transition '''
        oldspeed = self.state[1]
        if action == 1:
			self.state[1] = min(self.state[1]+self.acc, self.speedlimit)
        elif action == -1:
			self.state[1] = max(self.state[1]-self.acc, -self.speedlimit)
        else:
            print('illegal action!')
        self.state[0] += (self.state[1]+oldspeed)/2./self.freq
        reward = -abs(self.target-self.state[0])
        if self.state[0] >= self.room_size:
		    self.state[0] = self.room_size 
		    self.state[1] = 0
		    reward = -1000000
        if self.state[0] <= -self.room_size:
            self.state[0] = -self.room_size 
            self.state[1] = 0
            reward = -1000000
        #reward = -abs(self.target-self.state[0])
        if abs(self.state[0] - self.target) < 0.5 and self.state[2]!=1:# and (self.state[0] - self.target) < 1 and self.state[1] == 0:
			self.state[2] = 1
			reward=1000*self.state[1]
        if self.state[2] == 1:# and abs(self.state[1]) < 0.2:
        	'''if reward !=-1000000:
        	    reward=1000'''
        	return self.state, reward, True
        else:
            return self.state, reward, False


'''
TurtleBot Agent

Description:
Class, called by SI, defining agent's behavior & knowledge, SARSA-lambda
'''
class TB(object):

    '''turtlebot's knowledge'''
    epsilon = 0.3# how greedy tb is
    alpha = 1e-2 # learning rate
    gamma = 0.7 # discount factor
    l = 0.02 # lambda for SARSA(l)
    
    state_dim = 3. # 3-dim continuous state for task 1
    action_num = 2.  # discrete actions of -1/1
    state = None
    eli1 = None # eligibility trace
    #eli2 = None
    action = 0.
    
    '''func approximator'''
    q_w1 = None # 2nd order linear func parameter
    q_w2 = None	
    def qFunc(self, s, a, derivative = False):
        sa = np.concatenate((s,[a]))        
        #s_ex = np.concatenate(([sa[0]*sa[1]],[sa[0]*sa[2]],[sa[0]*sa[3]],[sa[1]*sa[2]],[sa[1]*sa[3]],[sa[2]*sa[3]],np.power(sa,2),sa,[1.]))
        speedlimit = 0.5
        roomsize = 3
        tiling_number = 2
        resolution = 10
    	speedbin1 = np.linspace(-speedlimit-0.5, speedlimit+0.5, num=resolution+1)
    	speedbin2 = np.linspace(-speedlimit-0.4, speedlimit+0.6, num=resolution+1)
    	tile1 = np.digitize(s[1],speedbin1,right=False)
    	tile2 = np.digitize(s[1],speedbin2,right=False)
    	posbin1 = np.linspace(-roomsize-0.5, roomsize+0.5, num=resolution+1)
    	posbin2 = np.linspace(-roomsize-0.4, roomsize+0.6, num=resolution+1)
    	tile3 = np.digitize(s[0],posbin1 ,right=False)
    	tile4 = np.digitize(s[0],posbin2 ,right=False)
    	tilex = np.digitize(-3,posbin1 ,right=False)
    	ans = np.zeros(400)
    	if a==1:
    	    ans[10*tile1+tile3-10]=1
    	    ans[90+10*tile2+tile4]=1
    	else:
    	    ans[10*tile1+tile3-10+200]=1
    	    ans[90+10*tile2+tile4+200]=1
        s_ex = ans
        
        
        
        
        #s_ex = np.concatenate((sa,[1.])) 
        #sa = [sa, sa]
        #poly = PolynomialFeatures(3)
        #s_ex = poly.fit_transform(sa)
        #s_ex = s_ex[0]
        if derivative:
            return s_ex
        #if s[2] == 0:
            #return np.sum(self.q_w1*s_ex)	
        #elif s[2] == 1:
        return np.sum(self.q_w1*s_ex)
        
    
    	
    '''take actions'''
    def actGreedy(self):
        q1 = self.qFunc(self.state, -1)
        q2 = self.qFunc(self.state,  1)
        if q1 > q2:  # take action w.r.t max Q-value
            return -1
        elif q1 < q2:
            return 1
        else:
            return self.actRandom()
        
    def actRandom(self):
        return 2*int(random.random()*self.action_num)-1 # take random action
        
    '''run 1 step w.r.t. given observations'''
    def runSARSA(self, s, r=0, start=False, done=False):
        if start:
            self.state = s; self.action = self.actGreedy(); return self.action
        if random.random()>self.epsilon: # epsilon greedy
            a = self.actGreedy()
        else:
            a = self.actRandom()
        dt = self.gamma*self.qFunc(s,a) + r - self.qFunc(self.state,self.action)
        if self.state[2] == 0 and s[2] == 0:
            self.eli1 = self.gamma * self.l * self.eli1 + \
                self.qFunc(self.state, self.action, derivative=True) # update eligibility trace
            self.q_w1 += self.alpha * dt * self.eli1 # update func parameter
        elif self.state[2] == 1 and s[2] == 1:
            self.eli2 = self.gamma * self.l * self.eli2 + \
                self.qFunc(self.state, self.action, derivative=True) # update eligibility trace
            self.q_w2 += self.alpha * dt * self.eli2 # update func parameter
        self.state = s; self.action = a; return a
    def __init__(self):
        self.state = np.zeros(self.state_dim)
        self.eli1 = np.zeros(400)#pow(2,self.state_dim) + 1)
        self.q_w1 = np.zeros(400)
        self.eli2 = np.zeros(400)
        self.q_w2 = np.zeros(400)
        


if __name__ == '__main__':
    env = ENV()
    agent = TB()
    episodes = list()
    for i_episode in range(2000):
        trace1 = list()
        trace2 = list()
        state = env.reset()
        action = agent.runSARSA(state, start=True)
        for t in range(1000):
            state, reward, done = env.step(action)
            trace1.append(state[0])
            trace2.append(state[1])
            action = agent.runSARSA(state, reward)
            if done:
                break
        episodes.append(t+1)
        print("{}th episode finished after {} timesteps, Q value at start {}, {}".format(i_episode, t+1, \
            agent.qFunc([0,0,0],-1), agent.qFunc([env.target-0.1,0,0],1),agent.q_w1))
    plt.plot(trace1)
    plt.plot(trace2)
    plt.show()
    plt.plot(episodes)	
    plt.show()
            
            



