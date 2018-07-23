#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 00:38:11 2018

@author: vipin
"""


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing dataset
dataset = pd.read_csv('dataset.csv')

x = dataset.iloc[:,:].values

#using elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters=i , init='k-means++', max_iter=300,n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel("number of clusters")
plt.ylabel("wcss")
plt.show()
#applying correct no of clusters
kmeans = KMeans(n_clusters= 5, init='k-means++', max_iter=10, n_init=10, random_state=0)
y_means = kmeans.fit_predict(x)
plt.scatter(x[y_means == 0,0],x[y_means == 0,1], s=100, c='red', label='cluster 0')
plt.scatter(x[y_means == 1,0],x[y_means == 1,1], s=100, c='green', label='cluster 1')
plt.scatter(x[y_means == 2,0],x[y_means == 2,1], s=100, c='yellow', label='cluster 2')
plt.scatter(x[y_means == 3,0],x[y_means == 3,1], s=100, c='blue', label='cluster 3')
plt.scatter(x[y_means == 4,0],x[y_means == 4,1], s=100, c='pink', label='cluster 4')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='grey', label='centroids')
plt.title("survey")
plt.xlabel("humidity")
plt.ylabel("temperature")
plt.legend()
plt.show()
cen_x=kmeans.cluster_centers_[:,0]
cen_y=kmeans.cluster_centers_[:,1]
# R matrix

R = np.matrix([ [-1,0,-1,-1,-1],
		[0,-1,0,0,-1],
		[-1,0,-1,-1,100],
		[-1,0,-1,-1,100],
		[-1,-1,0,0,100]])
# Q matrix

Q = np.zeros([5,5])
# Gamma (learning parameter).
gamma = 0.8

# Initial state. (Usually to be chosen at random)
initial_state = 1

# This function returns all available actions in the state given as an argument
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

# Get available actions in the current state
available_act = available_actions(initial_state) 

# This function chooses at random which action to be performed within the range 
# of all the available actions.
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act,1))
    return next_action

# Sample next action to be performed
action = sample_next_action(available_act)

# This function updates the Q matrix according to the path selected and the Q 
# learning algorithm
def update(current_state, action, gamma):
    max = np.max(Q[action,])
    
    # Q learning formula
    Q[current_state, action] = R[current_state, action] + gamma * max
     
     
update(initial_state,action,gamma)
print(Q)


#-------------------------------------------------------------------------------
# Training

# Train over 10 000 iterations. (Re-iterate the process above).
for i in range(10000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state,action,gamma)
    
# Normalize the "trained" Q matrix
print("Trained Q matrix:")
print(Q)
print("Q matrix when divided by max of Q = 500")
print(Q/np.max(Q))
print("Q matrix values between 0 to 100")
print(Q/np.max(Q)*100)



#-------------------------------------------------------------------------------
# Testing
# Goal state = 4
# Best sequence path starting from 2 -> 2
cost = 0
current_state = 2
steps = [current_state]
while current_state != 4:
     max2 =  np.max(Q[current_state,])
     next_step_index = np.argmax(Q[current_state,] )
     cost = cost+Q[current_state,next_step_index]
    
     
     
    
     steps.append(next_step_index)
     current_state = next_step_index 

# Print selected sequence of steps
print("Selected path:")
print(steps)

print("total cost :")
print(cost)

print (Q)
    
x_plot = cen_x=kmeans.cluster_centers_[steps,0]
y_plot = cen_x=kmeans.cluster_centers_[steps,1]
plt.plot(x_plot,y_plot)
plt.show()