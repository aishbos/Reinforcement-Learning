import numpy as np

import util
from agent import Agent


# TASK 3

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.3):
        """ A Q-Learning agent gets nothing about the mdp on construction other than a function mapping states to
        actions. The other parameters govern its exploration strategy and learning rate. """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        self.qInitValue = 0  # initial value for states
        self.Q = {}

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    def getValue(self, state):
        """ Look up the current value of the state. """
        # *********
        # TODO 3.1.
        if state not in self.Q : #if state not yet visited, return 0
            return 0.0
        else : #else return the maximum among the value functions
            V = -np.inf
            for action in self.Q[state] : 
                if self.Q[state][action] > V : 
                    V = self.Q[state][action]
            return V
        # *********

    def getQValue(self, state, action):
        """ Look up the current q-value of the state action pair. """
        # *********
        # TODO 3.2.
        if state not in self.Q :#if state not yet visited, return 0
            return 0.0
        else :
            return self.Q[state][action]
        # *********

    def getPolicy(self, state):
        """ Look up the current recommendation for the state. """
        # *********
        # TODO 3.3.
        if state not in self.Q  : #if state not visited, recommend random action
            return self.getRandomAction(state)
        else : #else return the action that maximises the Q Value
            max_Q = -np.inf
            max_action = None
            for action in self.Q[state] : 
                if self.Q[state][action] > max_Q : 
                    max_Q = self.Q[state][action]
                    max_action = action
            return max_action
        # *********

    def getRandomAction(self, state):
        all_actions = self.actionFunction(state)
        if len(all_actions) > 0: #if non empty action set, then return a random action from the set
            # *********
            return np.random.choice(all_actions)
            # *********
        else:
            return "exit"

    def getAction(self, state):
        """ Choose an action: this will require that your agent balance exploration and exploitation as appropriate. """
        # *********
        # TODO 3.4.
        if np.random.rand() < self.epsilon : #explore
            return self.getRandomAction(state)
        else: #exploit
            return self.getPolicy(state)
        # *********

    def update(self, state, action, nextState, reward):
        """ Update parameters in response to the observed transition. """
        # *********
        # TODO 3.5.
        if state not in self.Q : #if current state not yet visited, initialize
            self.Q[state] = {}
            actions = self.actionFunction(state)
            for a in actions : 
                self.Q[state][a] = 0.0 #set all to zero
                
        if nextState not in self.Q: #if next state not yet visited, its QValue is zero
            max_Q = 0.0
        else : #else, find maximum of QValue
            Qs = self.Q[nextState]
            max_Q = -np.inf
            for nextAction in Qs : 
                if Qs[nextAction] > max_Q : 
                    max_Q = Qs[nextAction]
        
        self.Q[state][action] = self.Q[state][action] + self.learningRate * (reward + self.discount * max_Q - self.Q[state][action]) 
        # *********
