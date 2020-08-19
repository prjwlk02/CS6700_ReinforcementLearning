"""SOFTMAX_Algorithm"""

#Importing the librarires
import numpy as np
import matplotlib.pyplot as plt
import time


"""Setting up the 10 armed bandit testbed for 2000 bandits"""

bandits = 2000                                                          #No of machines
arms = 10                                                               #lever/action in each machine
runs = 1000                                                             #No of pulls

T = [[0.01,'r'],[0.02,'g'],[0.1,'b'],[1,'k']]

true_values = np.random.randn(bandits, arms)

Average_Rewards = []
opt_arm_percent_T=[]

#ALGORITHM

for t in T:

  start=time.clock()
  totl_time=0.0
  opt_arm_count=[]

  Rewards_run_bandits = []
  for i in range(bandits):                                           #Running the Softmax algorithm for 2000 bandit problems separately
    
    opt_arms = []
    Reward_estimate = np.random.normal(true_values[i], 1)            #Initialising the estimates values of 10 arms of each bandit in each run
    count = [1]*10                                                   #Picking up each arm once



    Rewards_run = [0] 
    for k in range(1,runs):

      Prob = np.array(list(map(np.exp, np.array(Reward_estimate)/t[0])))  #Applying the softmax function with Gibbs equation to arms
      Prob = Prob/np.sum(Prob)                                            #Calculating the probabilities
    
      action = int(np.random.choice(list(range(arms)), 1, p=Prob))        #Choosing the arm based on probability 
      Reward = np.random.normal(true_values[i][action], 1)                #Estimate value of reward for the chosen action from respective distribution

      Reward_estimate[action] = (Reward_estimate[action]*count[action] + Reward)/(count[action]+1) # Updating the estimate of that action
      if action == np.argmax(true_values[i]):
        opt_arms.append(1)
      else:
        opt_arms.append(0)
      count[action]+=1
      Rewards_run.append(Reward)                                          #Saving the rewards of chosen action for a particular run for 1 bandit and 
                                                                          # finally 1000 runs for 1 bandit
    opt_arm_count.append(opt_arms)  
    Rewards_run_bandits.append(Rewards_run)                               #Saving the rewards of 1000 runs of each bandit 

  Rewards_run_bandits = np.array(Rewards_run_bandits)                     #Converting to numpy array
  Average_Rewards.append(np.mean(Rewards_run_bandits,axis = 0))            #Average of rewards of all 2000 bandits at all runs
  opt_arm_percent_T.append(np.mean(opt_arm_count,axis=0)*100)
  end = time.clock()
  totl_time = -start+end
  print('Total_computing_time for t = %f is %f' %(t[0], totl_time))

#Plotting the perfomance of algorithm over runs

for temp in range(len(T)):
  plt.plot(list(range(1000)), Average_Rewards[temp], c = T[temp][1], label = f'T = {T[temp][0]}') #Plotting the average estimate of 2000 bandits at each run
plt.legend()
plt.xlabel('Runs')
plt.ylabel('Average Reward')
plt.title('SOFTMAX_Algorithm')
plt.show()


for temp in range(len(T)):
  plt.plot(list(range(999)), opt_arm_percent_T[temp] , label = f'T = {T[temp][0]}', c = T[temp][1])
plt.ylabel('Optimal Arms Percentage')
plt.xlabel('Runs')
plt.title('SOFTMAX_ Algorithm')
plt.legend()
plt.show()