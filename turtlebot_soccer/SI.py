#!/usr/bin/env python

import rospy
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

'''
TurtleBot Agent

Description:
Class, called by SI, defining agent's behavior & knowledge, SARSA-lambda
'''
class TB(object):
    
    '''turtlebot's knowledge'''
    epsilon = 0.3 # how greedy tb is
    alpha = 0.01 # learning rate
    gamma = 0.9 # discount factor
    l = 0.7 # lambda for SARSA(l)
    
    state_dim = 3. # 3-dim continuous state for task 1
    action_num = 2.  # discrete actions of 0/1
    state = np.zeros(self.state_dim)
    eli = np.zeros(self.state_dim) # eligibility trace
    action = 0.
    
    '''func approximator'''
    q_w = np.zeros((self.action_num, self.state_dim)) # 2nd order linear func parameter
    def qFunc(self, s, derivative = False):
        s_ex = np.concatenate((np.power(s,2),s,[1.])) # the feature vector
        if derivative:
            return s_ex
        return np.sum(self.q_w*s_ex, axis=1)

    '''take actions'''
    def actGreedy(self):
        return np.argmax(self.qFunc(self.state)) # take action w.r.t max Q-value

    def actRandom(self):
        return int(random.random()*self.action_num) # take random action
        
    '''run 1 step w.r.t. given observations'''
    def runSARSA(self, s, r=0, start=False, done=False):
        if start:
            self.state = s; self.action = self.actRandom; return self.action
        if random.random()>self.epsilon: # epsilon greedy
            a = self.actGreedy()
        else:
            a = self.actRandom()
        dt = self.gamma*self.qFunc(s)[a] + r - self.qFunc(self.state)[self.action]
        self.eli = self.gamma * self.l * self.eli + self.qFunc(self.state, derivative=True) # update eligibility trace
        self.q_w += self.alpha * dt * self.eli # update func parameter
        self.state = s; self.action = a; return a
    
    def __init__(self):
        # nothing needed


'''
Simulation Interface

Description:
ROS node, handling real-time information about the ball

Publishing:
ball_pose: pose of the ball, [X, Y, theta]: position + orientation
ball_vel: linear instantaneous velocity of the ball, [X, Y]
'''
class SI(object):
	''' Data for the Turtlebot'''
	'''
	turtlebot_pose = Pose2D() # TODO: PoseWithCovariance
	turtlebot_vel = Pose2D()
	turtlebot_acc = Accel() # TODO: AccelWithCovariance
	'''
	
	# pose of turtle bot
	tb_pos = (0,0,0)
	tb_ang = Quaternion()
	
	# left and right wheel
	angLeft = 0.0 # positive direction: clockwise: Left - Right
	angRight = 0.0
	angAbs = 0.0 # Absolute rotation angle
	
	# Velocity of TB
	tb_vel = Pose2D() # Pose2D: float64 x, float64 y, float64 theta
	freq = 10.0 # Frequency of updating velocity: 10Hz
	posX = [0.0]*2 # frequency of /tf: 60Hz 
	posY = [0.0]*2 # [old, new]
	posT = [0.0]*2 # for angular velocity
	
	# Command (to publish)
	tb_cmd = Pose2D() # X_vel, Y_vel, anguler_vel
	
	# TODO: PoseArray: Header header, Pose[] poses

	''' Data for the Ball '''
	ball_pose = Pose2D()
	ball_vel = Pose2D()
	ball_acc = Accel()

	''' Data for the Gate '''
	gate_pose = Pose2D()
	gate_vel = Pose2D()
	gate_acc = Accel()

	# Initialization Function
	def __init__(self):
		
		''' Subscriber '''
		self.subJointStates = rospy.Subscriber("/joint_states", JointState, self.cbJointStates)
		# tf message need to be treated differently
		self.tf = tf.TransformListener()
		
		# Timer: Update posX and posY for calculating velocity
		rospy.Timer(rospy.Duration(1/(self.freq)), self.cbTimerVel, oneshot=False)
		
		# TODO: More subscribers
		
		# Publisher
		self.pubTBCommand = rospy.Publisher("TBCommand", Pose2D, queue_size = 100)
		


	'''Callback Functions'''
	def cbJointStates(self, data):
	    self.angLeft = data.position[0]
	    self.angRight = data.position[1]
	    self.angAbs = (self.angLeft - self.angRight)%(2*180)
	    if self.angAbs > 180:
	        self.angAbs -= 2*180
	        # TODO: Radian to Degree
	        
	def cbTimerVel(self, event):
		# update
		self.posX[1] = self.tb_pos[0]; self.posY[1] = self.tb_pos[1]; self.posT[1] = self.angAbs
		# calculate
		self.tb_vel.x = (self.posX[1] - self.posX[0])/(1/self.freq)
		self.tb_vel.y = (self.posY[1] - self.posY[0])/(1/self.freq)
		self.tb_vel.theta = (self.posT[1] - self.posT[0])/(1/self.freq)
		self.posX[0] = self.posX[1]; self.posY[0] = self.posY[1]; self.posT[0] = self.posT[1]

	'''Signal Processing Functions'''
	def _run(self):
		while not rospy.is_shutdown():
		    '''
		    self.tb_cmd.x = 0.0
		    self.tb_cmd.y = 0.0
		    self.tb_cmd.theta = 0.0
		    '''
		    if self.tf.frameExists("/odom") and self.tf.frameExists("/base_footprint"):
			t = self.tf.getLatestCommonTime("/odom", "/base_footprint")
			self.tb_pos = self.tf.lookupTransform("/odom", "/base_footprint", t)[0]
		    
		    self.pubTBCommand.publish(self.tb_vel)
		    #rospy.spin()

if __name__ == '__main__':
	try:
		rospy.init_node('Simulation_Interface', anonymous=True)
		si = SI()
		si._run()
	except rospy.ROSInterruptException:
		raise NameError('I Know Nothing')
