"""UCB1_ALGORITHM"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import time

"""Setting up the 10 armed bandit testbed for 2000 bandits"""

bandits  = 2000                                                             #No of machines
arms = 10                                                                   #lever/action in each machine
runs = 1000                                                                 #No of pulls

true_values = np.random.randn(bandits, arms)                                #Setting up mean values for reward distribution for each action in all bandits

Rewards_run_bandits = []
c = [[0.1,'r'], [2,'g'], [5,'k']]   #Degree of Exploration
#ALGORITHM
opt_arm_percent_c=[]
for d in c:

  start=time.clock()
  totl_time=0.0
  opt_arm_count=[]

  for i in range(bandits):
    opt_arms=[]
    Rewards_run = []
    Reward_estimate = np.random.normal(true_values[i], 1)                    #Estimate value of reward from their distribution
    count = list(np.ones(arms))                                              #Selectiong all arms once at start
    bands = np.array(Reward_estimate[:]) + (d[0]*np.log(arms)/count[1])**0.5 #Updating the bands of all arms cosen once at start
    Rewards_run = Rewards_run + list(bands) 

    #Now, choosing best arms from the algorithm
    for k in range(arms, runs+arms):
      
      action = np.argmax(bands) #Best action

      if action == np.argmax(true_values[i]):
        opt_arms.append(1)                                                   #Storing 1 if best action is chosen
      else:
        opt_arms.append(0)

      Imm_Reward = np.random.normal(true_values[i][action], 1)                                         #Reward of the best action
      Reward_estimate[action] = (Reward_estimate[action]*count[action] + Imm_Reward)/(count[action]+1) #Updating the estimate of reward
      count[action]+=1                                                                                 #Updating the count of chosen action

      """ updating the bands for each run"""

      for l in range(arms):
        bands[l] = Reward_estimate[l] + (d[0]*np.log(k)/count[l])**0.5                #updating the bands according to chosen arms
      Rewards_run.append(Imm_Reward)                                                  #Saving the rewards of chosen action for a particular run for 1 bandit and
                                                                                      # and finally 1000 runs for 1 bandit
    Rewards_run_bandits.append(Rewards_run)                                           #Saving the rewards of 1000 runs of each bandit
    opt_arm_count.append(opt_arms)                                                    #Storing no of times the best action is chosen in all bandits
  Average_Rewards = np.mean(Rewards_run_bandits, axis = 0)[arms:]                     #Average of rewards of all 2000 bandits at all runs
  opt_arm_count = np.mean(opt_arm_count, axis = 0)*100                                #Optimal arms percentage

  opt_arm_percent_c.append(opt_arm_count)                                            

  end = time.clock()
  totl_time = -start+end
  print('Total_computing_time for ddd = %f is %f' %(d[0], totl_time))
#Plotting the perfomance of algorithm over runs

  plt.plot(list(range(1000)), Average_Rewards, label = f'c = {d[0]}',c=d[1]) #Plotting the average estimate of 2000 bandits at each run
plt.legend()
plt.xlabel('Runs')
plt.ylabel('Average Reward')
plt.title('UCB1')
plt.show()

for d in range(len(c)):
  plt.plot(list(range(1000)), opt_arm_percent_c[d] , label = f'c = {c[d][0]}', c = c[d][1])
plt.ylabel('Optimal Arms Percentage')
plt.xlabel('Runs')
plt.title('UCB1 algorithm')
plt.legend()
plt.show()


