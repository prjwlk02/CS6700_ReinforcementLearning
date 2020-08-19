""" EPSILON_GREEDY_Algorithm"""

# Importing the libraries
import random
import numpy as np
import matplotlib.pyplot as plt
import time

"""Setting up the 10 armed bandit testbed for 2000 bandits"""

bandits = 2000                                                                  #No of machines
arms = 10                                                                       #lever/action in each machine
runs = 1000                                                                     #No of pulls

epsilon = [[0,'k'],[0.1,'r'],[0.2,'g'],[0.3,'b'],[1,'m']]                       #Different epsilon values
true_values = []
for k in range(bandits):
  tv = {}
  for i in range(arms):
    tv[f'arm{i}'] = np.random.randn()                                           #Setting up mean values for reward distribution for each action in all bandits
  true_values.append(tv)


#ALGORITHM
opt_arm_percent_eps=[]

for eps in epsilon:

  start=time.clock()
  totl_time=0.0

  Rewards_run_bandits = []
  opt_arm_count = []

  for k in range(bandits):

    opt_arms = []
    Rewards_run = []
    Reward_estimate = {}

    for i in range(arms):
        Reward_estimate[f'arm{i}'] = float(np.random.normal(true_values[k][f'arm{i}'], 1, 1)) #Estimate value of reward from their distribution
    a_star = np.argmax(list(Reward_estimate.values()))                                        #Best action for a bandit

    count = {}
    for j in range(arms):
        count[f'arm{j}'] = 1                                            #Each arm of all bandits are selected for once at start
    
    for l in range(runs):
        a_star = np.argmax(list(Reward_estimate.values()))              #Best action for a bandit for the particular run
        a = np.random.random()                                          #No. taken from an uniform distribution of 0 to 1 randomly
        
        if a > eps[0]:                                                  #No. compared to the epsilon values
            action = f'arm{a_star}'                                     #Exploitation ; Best arm till now is chosen
        else:
            action = random.choice(list(Reward_estimate.keys()))        #Exploration ; Any arm is chosen randomly

        if action == f'arm{np.argmax(list(true_values[k].values()))}':
          opt_arms.append(1)                                            #1 if optimal arm is chosen
        else:
          opt_arms.append(0)


        Imm_Reward = np.random.normal(true_values[k][action], 1)       #Reward is generated based on chosen action from the reward distribution
        Rewards_run.append(Imm_Reward)                                 #Saving the rewards of chosen action for a particular run for 1 bandit and 
                                                                       #finally 1000 runs for 1 bandit

        Reward_estimate[action] = (Reward_estimate[action]*count[action] + Imm_Reward)/(count[action]+1) #Updating the reward estimate
        count[action]+=1                                                                                 #Stores the no of times one action is chosen

    opt_arm_count.append(np.array(opt_arms))                                                           #Count of optimal arms
    Rewards_run_bandits.append(np.array(Rewards_run))                                                  #Saving the rewards of 1000 runs of each bandit 
  
  Rewards_run_bandits = np.array(Rewards_run_bandits)                                                  #Converting to numpy array
  opt_arm_count = np.array(opt_arm_count)                                                              #Converting to numpy array

  Average_Rewards = np.mean(Rewards_run_bandits, axis = 0)                          #Average of rewards of all 2000 bandits at all runs
  opt_arm_percent = np.sum(opt_arm_count,axis = 0)*(100/bandits)                    #Percent of optimal arms chosen in total arms chosen in all bandits
  
  opt_arm_percent_eps.append(opt_arm_percent)

  end = time.clock()
  totl_time = -start+end
  print('Total_computing_time for ε = %f is %f' %(eps[0], totl_time))
#Plotting the perfomance of algorithm over runs


  plt.plot(list(range(1000)), Average_Rewards, label = f'ε = {eps[0]}', c = eps[1]) #Plotting the average estimate of 2000 bandits at each run
plt.ylabel('Average Reward')
plt.xlabel('Runs')
plt.title('EPSILON_GREEDY Algorithm')
plt.legend()
plt.show()

for eps in range(len(epsilon)):
  plt.plot(list(range(1000)), opt_arm_percent_eps[eps] , label = f'eps = {epsilon[eps][0]}', c = epsilon[eps][1])
plt.ylabel('Optimal Arms Percentage')
plt.xlabel('Runs')
plt.title('EPSILON_GREEDY Algorithm')
plt.legend()
plt.show()

