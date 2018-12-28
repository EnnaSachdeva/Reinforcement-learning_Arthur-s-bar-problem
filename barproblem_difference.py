#!/usr/bin/env python
from random import randrange, choice
#from Strategy import Strategy
from operator import itemgetter
import numpy as np
import random
#from environment import Env
from collections import defaultdict
import random
import pdb
import matplotlib.pyplot as plt
import math
import matplotlib
#import seaborn as sns; sns.set()

num_agents = 30
b = 5 # capacity/optimal number of people in the bar for each night
k = 7 # number of nights (in a week)
time_limit = 10000
epsilon = 0.1
epsilon_decay = 0.999
learning_rate = 0.1
code_running = 1

#class agents(object):

np.random.seed()
class agent:
    """Defining the Class for the agent going to the bar.

    This class will represent a agent for our El Farol Bar/Minority
    Game problem. Each agent will have a simulated memory and a simulated
    decision protocol.

    Attributes:
        name: A reference to name if ever needed.
        attend: A boolean value to determine whether the agent will
            attend the bar this week or not.
        strategies: Dictionary of the key values of strategies
        has_random: A boolean, True if random strategy is assigned
            else False
        random_dict: Dictionary to hold random strategy value and
            score for indivdual agent object
    """


    no_inst = 0   # initialize the random strategies to 0
    recent_memory = []

	#Randomize attendance of all agents memory
    #for i in range(0, num_agents):
	#	recent_memory.append(randrange(0,11))

    def __init__(self, agent_index):  # init constuctor
        """Initialize agent with a name for reference and dictionary for strategy keys"""

        super().__init__()
        # Increment the number of instances of people
        agent.no_inst += 1
        #self.name = agent_index
        self.strategies = {}
        self.index = agent_index
	    # If user receives random strategy set to true
        self.has_random = False
           # Default random strategy in the event it is needed
        self.random_dict = {"value": 0, "score": 0}
        # Set default of attending to false
        self.attend = False

    def select_night(self, agent_index, agent_value, epsilon):
        """each agent selects night randomly"""
        if np.random.rand() < epsilon:
            # take random action
            night_var = random.randint(0,k-1)
        else:
            # take action according to the q function table
            night_var = np.argmax(agent_value[agent_index])
        return night_var



    def get_system_reward(self, night_count):
        """estimate total number of agents that went there"""
        system_reward = 0
        for i in range(k): # for each night
            system_reward = system_reward + (night_count[i]*math.exp(-night_count[i]/b))
        return system_reward



    def get_local_reward(self, night_count):
        """estimate total number of agents that went there"""

        local_reward = night_count*math.exp(-night_count/b)
        return local_reward




    def get_reward_without1(self, night_count, agent_night):
        """estimate total number of agents that went there"""
        diff_reward = 0
        for i in range(k): # for each night
            if (i == agent_night):
                diff_reward = diff_reward + ((night_count[i]-1)*math.exp(-(night_count[i]-1)/b))
            else:
                diff_reward = diff_reward + (night_count[i]*math.exp(-night_count[i]/b))
        return diff_reward


    def get_reward_without2(self, night_count, agent_night):
        """estimate total number of agents that went there"""
        temp = random.randint(0,k-1)
        diff_reward = 0
        for i in range(k): # for each night
            if (i == temp):
                diff_reward = diff_reward + ((night_count[i]+1)*math.exp(-(night_count[i]+1)/b))
            elif (i == agent_night):
                diff_reward = diff_reward + ((night_count[i]-1)*math.exp(-(night_count[i]-1)/b))
            else:
                diff_reward = diff_reward + night_count[i]*math.exp(-night_count[i]/b)

        return diff_reward


    def get_new_local_reward(self, random_agent, night_count):
        """estimate total number of agents that went there"""
        new_loc_reward = (night_count+1)*math.exp(-(night_count+1)/b)
        return new_loc_reward

    def get_new_system_reward(self, agent_index, night_count, p, agent_action):
        """estimate total number of agents that went there"""
        new_system_reward = 0
        for i in range(k): # for each night
            if (i==agent_action):
                new_system_reward = new_system_reward + ((night_count[i]-1)*math.exp(-(night_count[i]-1)/b))
            elif (i==p):
                new_system_reward = new_system_reward + ((night_count[i]+1)*math.exp(-(night_count[i]+1)/b))
            else:
                new_system_reward = new_system_reward + (night_count[i]*math.exp(-(night_count[i]+1)/b))
        return new_system_reward



    def value_update(self, agentindex, agent_value, learning_rate, system_reward):
        """estimate total number of agents that went there"""
        agent_value = agent_value + learning_rate*(system_reward-agent_value)
        return agent_value



# Main function starts here
if __name__ == "__main__" :
    performance = [ [0] * time_limit for _ in range(code_running)]

    agent= agent(num_agents)
    for iter in range(code_running):
        agent_value = [ [0] * k for _ in range(num_agents)]
        global_reward =  [0] * time_limit
        print ("TRAINING GOING ON...")
        #factoredness_index = np.array([[[0]*k]*num_agents]*time_limit)
        factoredness_index = np.zeros((num_agents,k))
        #factoredness_index[1][0][0] = 5
        #pdb.set_trace()

        for weeks in range(time_limit):  # number of episodes
            agent_night = [ [0] * k for _ in range(num_agents)]
            #night =  np.zeros(num_agents)
            action =  [0] * num_agents
            #night_count = np.zeros(k)
            night_count = [0] * k
            agent_local_reward = [ [0] * k for _ in range(num_agents)]
            difference_reward1 = [ [0] * k for _ in range(num_agents)]
            difference_reward2 = [ [0] * k for _ in range(num_agents)]
            #print(agent_reward_estimate)

            #if (weeks + 1) % 10 == 0:

            #epsilon = epsilon * epsilon_decay
            #learning_rate = learning_rate*0.7
            for agentindex in range(num_agents):
                action[agentindex] = agent.select_night(agentindex, agent_value, epsilon) # night array stores the night corresponding to each agent
                temp = action[agentindex]
                #print("########", temp)
                #agent_night[agentindex][temp] = agent_night[agentindex][temp]+1
                night_count[int(temp)] = night_count[int(temp)]+1
                #print("########", night_count[int(temp)])


            # get system rewards based on number of agents in each night
            system_reward = agent.get_system_reward(night_count)
            #pdb.set_trace()
            global_reward[weeks] = system_reward
            # get local rewards
            for agentindex in range(num_agents):
                #pdb.set_trace()
                temp = night_count[action[agentindex]]
                agent_reward = agent.get_local_reward(temp) #get local reward based on counts of that night
                #pdb.set_trace()
                agent_local_reward[agentindex][action[agentindex]] = agent_reward  # storing agent reward in the agent
                #agent_local_reward[agentindex][action[agentindex]] = agent_reward  # storing agent reward in the agent

                #print(agentindex)
                #agent_value[agentindex][action[agentindex]] = agent_reward
                #difference_reward1[agentindex][action[agentindex]] = system_reward - agent.get_reward_without1(night_count, action[agentindex])
                difference_reward2[agentindex][action[agentindex]] = system_reward - agent.get_reward_without2(night_count, action[agentindex])

                #print("NIGHT COUNT:", night_count)
                #print(agent_reward_estimate)
                # Q values updates
                agent_value[agentindex][action[agentindex]] = agent.value_update(agentindex, agent_value[agentindex][action[agentindex]],
                                                                        learning_rate, difference_reward2[agentindex][action[agentindex]] )
            #print ("Starting agent and target locations", agent_state, target_state)
        #plotting graphs between the number of nights and the number of agents

        # if agents chooses the maximum of all estimates it gets
        #print(agent_value)


        '''
        # Determine factoredness
        # pick random agent
        random_agent_index = random.randint(0,num_agents-1)
        random_agent_action = action[random_agent_index]
        agent_state_reward = agent_local_reward[random_agent_index][random_agent_action]
        m = 0
        for i in range (k): # if agent took i night instead of what it already taken
            if (i!=random_agent_action):
                agent_new_reward = agent.get_new_local_reward(random_agent_index, night_count[i])
                system_new_reward = agent.get_new_system_reward(random_agent_index, night_count, i, random_agent_action)

                if ((agent_state_reward-agent_new_reward)*(system_reward-system_new_reward)>0) :
                    m = m+1

        factoredness_index[random_agent_index, random_agent_action] = float(m)/float((k-1))

    #pdb.set_trace()

    print(factoredness_index)
    '''

        nightcount = [0]*k

        # TESTING, final update
        for agentindex in range(num_agents):
            #print(agent_value[agentindex])
            temp = np.argmax(agent_value[agentindex])
            #print(temp)
            nightcount[temp] = nightcount[temp]+1

        print(nightcount)

        performance[iter] =  global_reward

    '''
    performance1 = [0] * time_limit
    #pdb.set_trace()
    for iter in range (code_running):
        performance1 = np.add(performance1, performance[iter])
    # plots for the rewards with each episode
    # line 1 points
    x1 = np.arange(time_limit)
    #pdb.set_trace()
    y1 = performance1/code_running
    e = np.std(performance, axis=0)
    #pdb.set_trace()
    #plt.plot(x1,y1,'--', color = 'b')

    plt.errorbar(x1, y1, e, linestyle='--', marker='o', mfc='red',mec='green' )
    #plt.plot(x1, y1, label = "Global reward w.r.t weeks")
    # plotting the line 2 points
    #plt.plot(x2, y2, label = "line 2")

    #print(x1)
    #print(y1)

    # naming the x axis
    plt.xlabel('weeks(episodes)')
    # naming the y axis
    plt.ylabel('Performance')
    # giving a title to my graph
    plt.title('Learning with Difference Reward!  with agents=40, K=5, b=4')

    # show a legend on the plot
    plt.legend(['Mean', 'Uncertainty'])
# function to show the plot
    plt.show()
    '''


    x = np.arange(k)
    y = nightcount
    plt.bar(x,y,align='center') # A bar chart
    plt.xlabel('nights')
    plt.ylabel('attendance')
    plt.show()
