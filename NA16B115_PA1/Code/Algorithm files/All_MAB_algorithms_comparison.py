# """MEDIAN_ELIMINATION_METHOD"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math

"""Setting up the 1000 armed bandit testbed for 2000 bandits"""

bandits = 2000 #no of machines
arms = 1000 #lever/action in each machine

true_values = np.random.normal(size=[bandits,arms]) #Making a reward distribution and creating a variable which gives 1 reward value from the distribution

# eps_delta_pairs=[[1.2,0.8,'g'],[0.6,0.4,'k'],[1.2,0.6,'r']] #Epsilon-Delta Pairs
eps_delta_pairs=[[1.2,0.6,'r']]

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
    
    Avg_Imm_Reward_l = Imm_Reward_l/int(smpl_no)

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
        if Avg_Imm_Reward_l[b][arm] >= Median_l[b]:
          true_val_new[b][arm_pos]=true_values_l[b][arm]
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

  # print('Median_computing_time for ε = %f, δ = %f is %f' %(eps_del[0], eps_del[1], times))
  # print('Total_computing_time for ε = %f, δ = %f is %f' %(eps_del[0],eps_del[1],totl_time))

Average_Rewards_median_elimination = Average_Rewards
runs_median_elimination=runs



""" EPSILON_GREEDY_Algorithm"""


# Importing the libraries
import random
import numpy as np
import matplotlib.pyplot as plt
import time

"""Setting up the 1000 armed bandit testbed for 2000 bandits"""

bandits = 2000 #No of machines
arms = 1000 #lever/action in each machine
runs = runs_median_elimination #No of pulls

# epsilon = [[0,'k'],[0.1,'r'],[0.2,'g'],[0.3,'b'],[1,'m']] #Different epsilon values
epsilon=[[0.1,'r']]
true_values = []
for k in range(bandits):
  tv = {}
  for i in range(arms):
    tv[f'arm{i}'] = np.random.randn() #Setting up mean values for reward distribution for each action in all bandits
  true_values.append(tv)


#ALGORITHM

for eps in epsilon:

  start=time.clock()
  totl_time=0.0
  
  Rewards_run_bandits = []
  for k in range(bandits):
    Rewards_run = []
    Reward_estimate = {}

    for i in range(arms):
        Reward_estimate[f'arm{i}'] = float(np.random.normal(true_values[k][f'arm{i}'], 1, 1)) #Estimate value of reward from their distribution
    a_star = np.argmax(list(Reward_estimate.values())) #Best action for a bandit

    count = {}
    for j in range(arms):
        count[f'arm{j}'] = 1 #Each arm of all bandits are selected for once at start
    for l in range(runs):
        a_star = np.argmax(list(Reward_estimate.values())) #Best action for a bandit for the particular run
        a = np.random.uniform() #No. taken from an uniform distribution of 0 to 1 randomly
        if a > eps[0]: #No. compared to the epsilon values
            action = f'arm{a_star}' #Exploitation ; Best arm till now is chosen
        else:
            action = random.choice(list(Reward_estimate.keys())) #Exploration ; Any arm is chosen randomly

        Imm_Reward = np.random.normal(true_values[k][action], 1) #Reward is generated based on chosen action from the reward distribution
        Rewards_run.append(Imm_Reward) #Saving the rewards of chosen action for a particular run for 1 bandit and finally 1000 runs for 1 bandit

        Reward_estimate[action] = (Reward_estimate[action]*count[action] + Imm_Reward)/(count[action]+1) #Updating the reward estimate
        count[action]+=1 #Stores the no of times one action is chosen

    Rewards_run_bandits.append(np.array(Rewards_run)) #Saving the rewards of 1000 runs of each bandit 
  Rewards_run_bandits = np.array(Rewards_run_bandits) #Converting to numpy array
  Average_Rewards = np.mean(Rewards_run_bandits, axis = 0) #Average of rewards of all 2000 bandits at all runs
  end = time.clock()
  totl_time = -start+end
  # print('Total_computing_time for ε = %f is %f' %(eps[0], totl_time))

Average_Rewards_epsilon_greedy = Average_Rewards



"""SOFTMAX_Algorithm"""

#Importing the librarires
import numpy as np
import matplotlib.pyplot as plt
import time


"""Setting up the 1000 armed bandit testbed for 2000 bandits"""

bandits = 2000 #No of machines
arms = 1000 #lever/action in each machine
runs = runs_median_elimination #No of pulls

# T = [[0.01,'r'],[0.02,'g'],[0.1,'b'],[1,'k']]
T=[0.01,'r']
true_values = np.random.randn(bandits, arms)

Average_Rewards = []

#ALGORITHM
start=time.clock()
totl_time=0.0

Rewards_run_bandits = []
for i in range(bandits): #Running the Softmax algorithm for 2000 bandit problems separately
  Reward_estimate = np.random.normal(true_values[i], 1) #Initialising the estimates values of 10 arms of each bandit in each run
  count = [1]*arms #Picking up each arm once



  Rewards_run = [0] 
  for k in range(runs):

    Prob = np.array(list(map(np.exp, np.array(Reward_estimate)/T[0])))  #Applying the softmax function with Gibbs equation to arms
    Prob = Prob/np.sum(Prob)   #Calculating the probabilities
    
    action = int(np.random.choice(list(range(arms)), 1, p=Prob)) #Choosing the arm based on probability 
    Reward = np.random.normal(true_values[i][action], 1) #Estimate value of reward for the chosen action from respective distribution

    Reward_estimate[action] = (Reward_estimate[action]*count[action] + Reward)/(count[action]+1) # Updating the estimate of that action
    count[action]+=1
    Rewards_run.append(Reward) #Saving the rewards of chosen action for a particular run for 1 bandit and finally 1000 runs for 1 bandit
      
  Rewards_run_bandits.append(Rewards_run) #Saving the rewards of 1000 runs of each bandit 

Rewards_run_bandits = np.array(Rewards_run_bandits) #Converting to numpy array
Average_Rewards = (np.mean(Rewards_run_bandits,axis = 0)) #Average of rewards of all 2000 bandits at all runs

end = time.clock()
Totl_time = -start+end
  # print('Total_computing_time for t = %f is %f' %(t[0], totl_time))

Average_Rewards_softmax = Average_Rewards 



"""UCB1_ALGORITHM"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import time

"""Setting up the 1000 armed bandit testbed for 2000 bandits"""

bandits  = 2000                                                             #No of machines
arms = 1000                                                                   #lever/action in each machine
runs = runs_median_elimination                                                                #No of pulls

true_values = np.random.randn(bandits, arms)                                #Setting up mean values for reward distribution for each action in all bandits

Rewards_run_bandits = []
c = [[2,'g']]  #Degree of Exploration
#ALGORITHM
# opt_arm_percent_c=[]
for d in c:

  start=time.clock()
  totl_time=0.0
  # opt_arm_count=[]

  for i in range(bandits):
    # opt_arms=[]
    Rewards_run = []
    Reward_estimate = np.random.normal(true_values[i], 1)                    #Estimate value of reward from their distribution
    count = list(np.ones(arms))                                              #Selectiong all arms once at start
    bands = np.array(Reward_estimate[:]) + (d[0]*np.log(arms)/count[1])**0.5 #Updating the bands of all arms cosen once at start
    Rewards_run = Rewards_run + list(bands) 

    #Now, choosing best arms from the algorithm
    for k in range(arms, runs+arms):
      
      action = np.argmax(bands) #Best action

      # if action == np.argmax(true_values[i]):
      #   opt_arms.append(1)                                                   #Storing 1 if best action is chosen
      # else:
      #   opt_arms.append(0)

      Imm_Reward = np.random.normal(true_values[i][action], 1)                                         #Reward of the best action
      Reward_estimate[action] = (Reward_estimate[action]*count[action] + Imm_Reward)/(count[action]+1) #Updating the estimate of reward
      count[action]+=1                                                                                 #Updating the count of chosen action

      """ updating the bands for each run"""

      for l in range(arms):
        bands[l] = Reward_estimate[l] + (d[0]*np.log(k)/count[l])**0.5                #updating the bands according to chosen arms
      Rewards_run.append(Imm_Reward)                                                  #Saving the rewards of chosen action for a particular run for 1 bandit and
                                                                                      # and finally 1000 runs for 1 bandit
    Rewards_run_bandits.append(Rewards_run)                                           #Saving the rewards of 1000 runs of each bandit
    # opt_arm_count.append(opt_arms)                                                    #Storing no of times the best action is chosen in all bandits
  Average_Rewards = np.mean(Rewards_run_bandits, axis = 0)[arms:]                     #Average of rewards of all 2000 bandits at all runs
  # opt_arm_count = np.mean(opt_arm_count, axis = 0)*100                                #Optimal arms percentage

  # opt_arm_percent_c.append(opt_arm_count)                                            

  end = time.clock()
  totl_time = -start+end
  # print('Total_computing_time for ddd = %f is %f' %(d[0], totl_time))

Average_Rewards_UCB1 = Average_Rewards







#PLOTTING THE GRAPH SHOWING COMPARISON BETWEEN DIFFERENT ALGORITHMS

import matplotlib.pyplot as plt

plt.plot(list(range(runs_median_elimination)), Average_Rewards_epsilon_greedy, c = 'k',label = f'Epsilon_Greedy for ε = 0.1' )
plt.plot(list(range(runs_median_elimination+1)), Average_Rewards_softmax, c = 'b', label = f'Softmax for t = 0.01')
plt.plot(list(range(runs_median_elimination)), Average_Rewards_UCB1, c = 'r', label = f'UCB1 for c = 2')
plt.plot(list(range(runs_median_elimination)), Average_Rewards_median_elimination, c = 'g', label = f'ε = 1.2, δ = 0.6')
plt.legend()
plt.title('Comparison between different Algorithms')
plt.xlabel('Runs')
plt.ylabel('Average Reward')