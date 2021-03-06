# -*- coding: utf-8 -*-
"""Rl_A1_P4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rmI-CNTWT_P5q01IPbxt2-wh9-W7gz0w
"""

"""MEDIAN_ELIMINATION_METHOD"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math

"""Setting up the 10 armed bandit testbed for 2000 bandits"""

bandits = 2000 #no of machines
arms = 10 #lever/action in each machine

true_values = np.random.normal(size=[bandits,arms]) #Making a reward distribution and creating a variable which gives 1 reward value from the distribution

eps_delta_pairs=[[1.2,0.8,'g'],[0.6,0.4,'k'],[1.2,0.6,'r']] #Epsilon-Delta Pairs

plt.figure(figsize = (10,5))
for eps_del in eps_delta_pairs:
  start=time.clock()
  totl_time=0.0
  times=0.0
  eps_frst = eps_del[0]
  delta_frst=eps_del[1]

  l=1     # first round
  eps_l = eps_frst/4.0
  delta_l = delta_frst/2.0

  true_values_l = true_values
  arms_l = arms

  runs = 0
  Average_Rewards=[]

  #ALGORITHM

  while(arms_l!=1):
    
    #Sampling each arm of bandits this no of times

    smpl_no = math.log10(3.0/delta_l)*4/(eps_l**2) 
    Imm_Reward_l = np.zeros((bandits,arms_l))


    for i in range(int(smpl_no)):
      Imm_Reward = np.random.normal(true_values_l,1) #Initialising the estimates values of 10 arms of each bandit in each run
      Imm_Reward_l = Imm_Reward_l + Imm_Reward #Taking the sum of all reward estimates for 'l'
      Average_Rewards.append(np.mean(Imm_Reward)) #Taking the mean of all reward estimates of all bandits in each run for plotting
      runs = runs + 1 #Keeping the tracks of no of runs

    #Average reward estimate of each arm of all bandits
    
    Avg_Imm_Reward_l = Imm_Reward_l/int(smpl_no)  #Averaged reward for each action of all bandits

    #Median Calculation of reward estimates for each bandit
    start_median = time.clock() #Time measurement for median
    
    Median_l = np.median(Avg_Imm_Reward_l,axis = 1)

    end_median = time.clock()
    times = times + (end_median-start_median)

    #Elimination of arms having rewards estimates less than median

    true_val_new = np.zeros((bandits,(arms_l-int(arms_l/2))))

    #Updating the bandit-arm matrix with optimal values of greater than or equal to median
    #MEDIAN ELIMINATION
    for b in range(bandits):
      arm_pos=0
      for arm in range(arms_l):
        if Avg_Imm_Reward_l[b][arm] >= Median_l[b]: #Checking rewards estimates lesser than the median for a particular bandit
          true_val_new[b][arm_pos]=true_values_l[b][arm]  #Updating reward estimates values for next round
          arm_pos = arm_pos + 1
    
    #Updating the values
    true_values_l = true_val_new
    arms_l = arms_l-int(arms_l/2)
    eps_l = 3*eps_l/4.0
    delta_l = delta_l/2
    l=l+1

  #Getting Reward estimates of last run.  

  smpl_no = math.log10(3.0/delta_l)*4/(eps_l**2)
  Imm_Reward_l = np.zeros((bandits,arms_l))
  for i in range(int(smpl_no)):
    Imm_Reward = np.random.normal(true_values_l,1)
    Imm_Reward_l = Imm_Reward_l + Imm_Reward
    Average_Rewards.append(np.mean(Imm_Reward))
    runs = runs + 1

  end = time.clock()
  totl_time = -start+end
  print('Median_computing_time for ε = %f, δ = %f is %f' %(eps_del[0], eps_del[1], times))
  print('Total_computing_time for ε = %f, δ = %f is %f' %(eps_del[0],eps_del[1],totl_time))

#Plotting the perfomance of algorithm over runs
  plt.plot(range(runs), Average_Rewards, eps_del[2],label = f'ε = {eps_del[0]}, δ ={eps_del[1]}',c = eps_del[2]) #Plotting the average estimate of 2000 bandits at each run
plt.xlabel('Runs')
plt.ylabel('Average Reward')
plt.title('MEDIAN_ELIMINATION Algorithm')
plt.legend()
plt.show()

