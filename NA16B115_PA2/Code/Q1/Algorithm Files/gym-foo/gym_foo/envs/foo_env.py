import gym
import numpy as np
from gym import error, spaces
import pandas as pd

class grid(gym.Env):
    metadata = {'render.modes':['Human']}
    
    def __init__(self):
        self.puddleworld = np.zeros([12,12])
        
        #Defining rewards of puddle in the grid
        self.puddleworld[2:7, [3, 8]] = -1
        self.puddleworld[6:9, [3, 7]] = -1
        self.puddleworld[[2, 8], 3:8] = -1
        
        #Puddles with -2 reward
        self.puddleworld[3:6, [4, 7]] = -2
        self.puddleworld[5:8, [4, 6]] = -2
        self.puddleworld[[3, 7], 4:7] = -2
        
        #Puddles with -3 reward
        self.puddleworld[4:7, 5] = -3
        self.puddleworld[4, 6] = -3
        
        #Actions
        up = [1,0]
        down = [-1,0]
        right = [0,1]
        left = [0,-1]
        self.actions = [up, down, right, left]
        
        self.action_space =  spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low = -3, high =10, shape = self.puddleworld.shape)
        
        #Terminal States
        self.terminal_states = [[0,11],[2,9],[6,7]]
        
        #Initial Reward
        self.Reward = 0
        
        #Start states
        self.start_states = [[5,0], [6,0], [10,0], [11,0]]
        
    #Deciding the terminal state 
    
    def set_goal(self, terminal):
        if terminal == 'A':
            self.puddleworld[self.terminal_states[0][0], self.terminal_states[0][1]] = 10
            self.puddleworld[2,9] = 0
            self.puddleworld[6,7] = -1
            self.wind = 1
            return self.terminal_states[0]
                    
        elif terminal == 'B':
            self.puddleworld[self.terminal_states[1][0], self.terminal_states[1][1]] = 10
            self.puddleworld[0,11] = 0
            self.puddleworld[6,7] = -1
            self.wind = 1
            return self.terminal_states[1]
                    
        elif terminal == 'C':
            self.puddleworld[self.terminal_states[2][0], self.terminal_states[2][1]] = 10
            self.puddleworld[2,9] = 0
            self.puddleworld[0,11] = 0
            self.wind = 0
            return self.terminal_states[2]
        
    #Updating reward given the current position
    
    def get_reward(self,position):
        self.reward = self.puddleworld[position[0],position[1]]
        return self.reward
    
    #If a particuar action is selected, we need to find out which action is really taken given different probabilities 
    
    def taken_action(self,selected_action):
        Prob_action = [0.1/3,0.1/3,0.1/3,0.1/3] #Giving all actions prob of 0.1/3
        Prob_action[selected_action]= 0.9 #Updating the prob of selected action as 0.9
        action = self.actions[np.random.choice(range(4),1,Prob_action)[0]]
        return action
    
    #Given current state and action to be taken in it, we find next state and reward from it. 
    
    def step(self, current_state, action):
        action = self.taken_action(action)
        
        #wind effect
        if self.wind:
            Prob_wind=[0.5,0.5]
            self.wind_effect = np.random.choice([0,1],1,Prob_wind)[0]
        else:
            self.wind_effect = 0
            
        #Checking boundary conditions
        #If agent is in corner states and action taken makes the agent move outside, its state won't change
        
        if(current_state[0]+action[0]<0 or 
           current_state[0]+action[0]>11 or 
           current_state[1]+action[1]+self.wind_effect<0 or 
           current_state[1]+action[1]+self.wind_effect>11):
            next_state = current_state
            self.reward = self.puddleworld[next_state[0],next_state[1]]
            return self.reward,next_state
        else:
            next_state = [current_state[0]+action[0],current_state[1]+action[1]+self.wind_effect]
            self.reward = self.get_reward(next_state)
            return self.reward,next_state
        
    #Action selection
    def random_action(self):
        #when agent is randomly taking any action
        Prob_action = [0.25,0.25,0.25,0.25]
        self.action = self.actions[np.random.choice(range(4),1,Prob_action)[0]]
        
    #Selection starting state with equal prob when game gets reset
    
    def reset(self):
        prob_start_states = [0.25,0.25,0.25,0.25]
        self.start_pos = self.start_states[np.random.choice(range(4),1,prob_start_states)[0]]
        return self.start_pos
    

    
    def render(self,state,mode='Human'):
        puddleworld_copy = self.puddleworld[:, :]
        puddleworld_copy = puddleworld_copy.astype('str')
    
        print(f'current_state: {state}')
        puddleworld_copy[state[0], state[1]] = 'Agent'
        print(pd.DataFrame(puddleworld_copy))
    def close(self):
        pass
