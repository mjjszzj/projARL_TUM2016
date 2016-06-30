#!/usr/bin/env python

import array
import serial
import struct
import binascii
import numpy as np
import math
import random
import matplotlib.pyplot as plt


class ENV(object):
    state_dim = 3
    action_num = 2
    state = None
    a_num = None
    q_w = None
    acc = 0.01
    freq = 1.
    target = 1.
    
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
            self.state[1] += self.acc
        elif action == -1:
            self.state[1] -= self.acc
        else:
            print('illegal action!')
        self.state[0] += (self.state[1]+oldspeed)/2./self.freq
        if self.state[0] <= 0 and self.state[1] < 0:
            self.state[0] = 0
            self.state[1] = 0
            reward -= 100000000
        elif self.state[0] >= 3 and self.state[1] >= 0:
            self.state[0] = 3
            self.state[1] = 0
            reward -= 100000000
        if self.state[0] > self.target and self.state[2] == 0:
            self.state[2] = 1 # two policy
            reward += 1000*self.state[1]
            #pass # one policy
        ''' reward function '''
        if self.state[2] == 0:
            reward += -abs(self.target-self.state[0])
        else:
            reward += -abs(self.state[1])
        #reward += -abs(self.target-self.state[0])-abs(self.state[1]/100)
        ''' done check '''
        if self.state[2] == 1 and self.state[1] < 0.005:
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
    epsilon = 0.3 # how greedy tb is
    alpha = 1e-9 # learning rate
    beta = 1e-9 # learning rate for second term
    gamma = 1 # discount factor
    l = 0.02 # lambda for SARSA(l)
    
    state_dim = 3. # 3-dim continuous state for task 1
    action_num = 2.  # discrete actions of -1/1
    state = None
    eli1 = None # eligibility trace
    eli2 = None
    action = 1.
    
    '''func approximator'''
    q_w1 = None # 2nd order linear func parameter
    q_w2 = None
    def qFunc(self, s, a, derivative = False):
        s1 = np.array(s[0:2])
        sa = np.concatenate((s1,[a],a*s1))
        s_ex = np.concatenate((np.power(sa,4),np.power(sa,3),np.power(sa,2),sa,[1.])) # the feature vector
        #print(a*s, sa,self.q_w1)
        if derivative:
            return s_ex
        if s[2] == 0:
            return np.sum(self.q_w1*s_ex)
        else:
            return np.sum(self.q_w2*s_ex)
        
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
    def runSARSA(self, s, a, r=0, start=False, done=False):
        if start:
            self.state = s; self.action = a; return
        dt = self.gamma*self.qFunc(s,a) + r - self.qFunc(self.state,self.action)
        ''' two policies '''
        
        if self.state[2] == 0 and s[2] == 0:
            #self.eli1 = np.minimum(self.gamma * self.l * self.eli1 + \
            #    self.qFunc(self.state, self.action, derivative=True), \
            #    10*np.ones(2*(2*self.state_dim + 1) + 1)) # update eligibility trace
            self.eli1 = self.gamma * self.l * self.eli1 + \
                self.qFunc(self.state, self.action, derivative=True)
            self.q_w1 += self.alpha * dt * self.eli1 # update func parameter
            #print(dt, self.eli1, self.q_w1)
        elif self.state[2] == 1 and s[2] == 1:
            self.eli2 = self.gamma * self.l * self.eli2 + \
                self.qFunc(self.state, self.action, derivative=True) # update eligibility trace
            self.q_w2 += self.beta * dt * self.eli2 # update func parameter
        
        ''' one policy '''
        '''
        self.eli1 = self.gamma * self.l * self.eli1 + \
                self.qFunc(self.state, self.action, derivative=True) # update eligibility trace
        self.q_w1 += self.beta * dt * self.eli1 # update func parameter
        '''
        self.state = s; self.action = a
    def __init__(self):
        self.state = np.zeros(self.state_dim)
        self.eli1 = np.zeros(4*(2*(self.state_dim-1) + 1) + 1)
        self.q_w1 = np.zeros(4*(2*(self.state_dim-1) + 1) + 1)
        self.eli2 = np.zeros(4*(2*(self.state_dim-1) + 1) + 1)
        self.q_w2 = np.zeros(4*(2*(self.state_dim-1) + 1) + 1)
        


if __name__ == '__main__':
    env = ENV()
    agent = TB()
    count = 0
    episodes = list()
    for i_episode in range(2000):
        trace1 = list()
        trace2 = list()
        state = env.reset()
        action = agent.actRandom()
        agent.runSARSA(state, action, start=True)
        for t in range(10000):
            count = count+1
            state, reward, done = env.step(action)
            trace1.append(state[0])
            trace2.append(state[1])
            if random.random() > (1./count):
                action = agent.actGreedy()
            else:
                action = agent.actRandom()
            agent.runSARSA(state, action, reward)
            #print(state, action, reward)
            #print(agent.qFunc([0,0,0],-1), agent.qFunc([0,0,0],1))
            if done:
                break
        episodes.append(t+1)
        print("{}th episode finished after {} timesteps, Q value at start {}, {}".format(i_episode, t+1, \
            agent.qFunc([0,0,0],-1), agent.qFunc([0,0,0],1)))
        plt.plot(trace1)
        plt.plot(trace2)
        plt.show()
    plt.plot(trace1)
    plt.plot(trace2)
    plt.show()
    plt.plot(episodes)
    plt.show()
            
            



