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
    state_dim = 6 #3#[x, y, velocity, theta, omega, goal_reached]
    action_dim = 2

    action_x_value = 1 # -1, 0, 1
    action_theta_value = 1# -1, 0, 1

    #state = [0]*6 # 1*6 Vector
    #a_num = [0]*2 # 1*2 Vector
    
    acc_x = 0.01
    acc_theta = 0.01

    freq = 10.
    target = [0.,-1.,0]#[0]*3# [x, y, omega]

    ball_radius = 0.2
    
    def __init__(self):
        self.state = np.zeros(self.state_dim)
        self.a_num = np.zeros(self.action_dim)
        
    def reset(self):
        self.state = np.zeros(self.state_dim)
        return self.state
        
    def step(self, action, start=False):
        reward = 0
        if start:
            return self.state, reward, False

        ''' state transition '''
        oldspeed_x = self.state[2]
        if action[0] == 1:
            self.state[2] = min(self.state[2]+self.acc_x, 0.3)
        elif action[0] == -1:
            self.state[2] = max(self.state[2]-self.acc_x, -0.3)
        elif action[0] == 0:
            pass
        else:
            print('illegal action in x direction!')

        oldspeed_theta = self.state[4]
        if action[1] == 1:
            self.state[4] = min(self.state[4]+self.acc_x, 0.6)
        elif action[1] == -1:
            self.state[4] = max(self.state[4]-self.acc_x, -0.6)
        elif action[1] == 0:
            pass
        else:
            print('illegal action in theta direction!')

        self.state[3] = np.mod(self.state[3]+(self.state[4]+oldspeed_theta)/2./self.freq, 2*math.pi)
        self.state[0] += (self.state[2]*np.cos(self.state[3])/self.freq)
        self.state[1] += (self.state[2]*np.sin(self.state[3])/self.freq)

        ''' border control '''
        x_min = 0; y_min = -1.5; x_max = 3; y_max = 1.5
        if self.state[0] <= x_min and self.state[2]*np.cos(self.state[3]) < 0:# check position in x direction
            self.state[0] = x_min
            self.state[2] = self.state[2]*np.sin(self.state[3])
            reward -= 10#0000
        elif self.state[1] <= y_min and self.state[2]*np.sin(self.state[3]) < 0:# check position in y direction
            self.state[1] = y_min
            self.state[2] = self.state[2]*np.cos(self.state[3])
            reward -= 10#0000
        elif self.state[0] >= x_max and self.state[2]*np.cos(self.state[3]) > 0:# check x-wall
            self.state[0] = x_max
            self.state[2] = self.state[2]*np.sin(self.state[3])
            reward -= 10#0000
        elif self.state[1] >= y_max and self.state[2]*np.sin(self.state[3]) > 0:# check y-wall
            self.state[1] = y_max
            self.state[2] = self.state[2]*np.cos(self.state[3])
            reward -= 10#0000
        
        ''' target check '''
        if np.power(self.state[0]-self.target[0], 2) + np.power(self.state[1]-self.target[1], 2) < np.power(self.ball_radius,2) \
                and self.state[5] == 0:
            self.state[5] = 1 # two policy
            print('target reached')
            reward += 100*self.state[2]
            #pass # one policy

        ''' reward function '''
        if self.state[5] == 0:
            #reward += -(np.power(self.state[0]-self.target[0], 2) + np.power(self.state[1]-self.target[1], 2))
            reward -= 1
        else:
            reward += -(abs(self.state[2]) + abs(self.state[4]))

        #reward += -abs(self.target-self.state[0])-abs(self.state[1]/100)
        ''' done check '''
        if self.state[5] == 1 :#and (abs(self.state[2]) + abs(self.state[4])) < 0.01:
            #reward += 500
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
    alpha = 1e-8 # learning rate 1e-6 1e-7
    beta = 1e-8 # learning rate for second term 1e-6 1e-7
    gamma = 1 # discount factor
    l = 1 # lambda for SARSA(l) 0.02
    
    state_dim = 6. # 3-dim continuous state for task 1
    action_num = 2.  # discrete actions of -1/1
    state = None
    eli1 = None # eligibility trace
    eli2 = None
    action = np.array([0, 0])
    
    '''func approximator'''
    q_w1 = None # 2nd order linear func parameter
    q_w2 = None
    def qFunc(self, s, a, derivative = False):
        s1 = np.array(s[0:5])
        sa = np.concatenate((s1,a,a[0]*s1,a[1]*s1))
        s_ex = np.concatenate((np.power(sa,4),np.power(sa,3),np.power(sa,2),sa,[1.])) # the feature vector
        #print(a*s, sa,self.q_w1)
        if derivative:
            return s_ex
        if s[5] == 0:
            return np.sum(self.q_w1*s_ex)
        else:
            return np.sum(self.q_w2*s_ex)
        
    '''take actions'''
    def actGreedy(self):
        possible_action=np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        vals = np.array([])
        for a in possible_action:
            vals = np.append(vals, self.qFunc(self.state, a))
        #print(vals)
        max_actions = possible_action[np.where(vals == max(vals))]
        if len(max_actions) == 1:
            return max_actions[0]
        else:
            return max_actions[int(random.random()*len(max_actions))]
        
    def actRandom(self):
        return np.array([2*int(random.random()*2)-1, \
                     2*int(random.random()*2)-1]) # take random action
        
    '''run 1 step w.r.t. given observations'''
    def runSARSA(self, s, a, r=0, start=False, done=False):
        if start:
            self.state = s; self.action = a; return
        dt = self.gamma*self.qFunc(s,a) + r - self.qFunc(self.state,self.action)
        ''' two policies '''
        
        if self.state[5] == 0 and s[5] == 0:
            #self.eli1 = np.minimum(self.gamma * self.l * self.eli1 + \
            #    self.qFunc(self.state, self.action, derivative=True), \
            #    10*np.ones(2*(2*self.state_dim + 1) + 1)) # update eligibility trace
            self.eli1 = self.gamma * self.l * self.eli1 + \
                self.qFunc(self.state, self.action, derivative=True)
            self.q_w1 += self.alpha * dt * self.eli1 # update func parameter
            #print(dt, self.eli1, self.q_w1)
        elif self.state[5] == 1 and s[5] == 1:
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
        self.eli1 = np.zeros(4*(self.state_dim-1+2+2*(self.state_dim-1)) + 1)
        self.q_w1 = np.zeros(4*(self.state_dim-1+2+2*(self.state_dim-1)) + 1)
        self.eli2 = np.zeros(4*(self.state_dim-1+2+2*(self.state_dim-1)) + 1)
        self.q_w2 = np.zeros(4*(self.state_dim-1+2+2*(self.state_dim-1)) + 1)
        


if __name__ == '__main__':
    env = ENV()
    agent = TB()
    count = 0
    episodes = list()
    for i_episode in range(1000):
        trace1 = list(); trace2 = list(); trace3 = list(); trace4 = list()
        state = env.reset()
        action = agent.actRandom()
        agent.runSARSA(state, action, start=True)
        for t in range(10000):
            count = count+1
            state, reward, done = env.step(action)
            #print(state, reward, action)
            trace1.append(state[0])
            trace2.append(state[1])
            trace3.append(state[2])
            trace4.append(state[4])
            if random.random() > (1./count):
                action = agent.actGreedy()
                #print(state, action, reward)
            else:
                action = agent.actRandom()
            agent.runSARSA(state, action, reward)   
            #print(agent.qFunc([0,0,0],-1), agent.qFunc([0,0,0],1))
            if np.mod(i_episode, 100) == 0:
                print(action)
            if done:
                break
        episodes.append(t+1)
        print("{}th episode finished after {} timesteps, Q value at start {}, {}".format(i_episode, t+1, \
            agent.qFunc([0,0,0,0,0,0],[1, 1]), agent.qFunc([0,0,0,0,0,0],[-1, 1])))
        if np.mod(i_episode, 100) == 0:
            plt.plot(trace1,trace2)
            plt.show()
            plt.plot(trace3)
            plt.plot(trace4)
            plt.show()
            print(agent.q_w1, agent.q_w2)
            plt.plot(episodes)
            plt.show()
    plt.plot(trace1)
    plt.plot(trace2)
    plt.show()
    plt.plot(episodes)
    plt.show()
            
            



