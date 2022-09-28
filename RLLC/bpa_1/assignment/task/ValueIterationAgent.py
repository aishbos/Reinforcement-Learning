from agent import Agent
import numpy as np

# TASK 2
class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # *************
        #  TODO 2.1 a)
        self.V = {state : 0 for state in states}
        # ************
        first_change_flag = False # for checking when was the first time the start state became non-zero
        for i in range(iterations):
            newV = {}
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                # **************
                # TODO 2.1. b)
                if self.mdp.isTerminal(s):
                    newV[s] = 0
                
                else: 
                    newV[s]=-np.inf
                    for action in actions:
                        q = self.getQValue(s,action)
                        if q>newV[s]:
                            newV[s] = q
            # check if start state is non-zero and if this is the first time it became non-zero
            if newV[self.mdp.getStartState()] != 0 and not first_change_flag:
                first_change_flag = True
                print("Start state " +str(self.mdp.getStartState())+" has non zero value " +
                      str(newV[self.mdp.getStartState()])+" after "+str(i)+ " rounds of value iteration")
            # have the values converged?
            if newV==self.V:
                self.V = newV
                print("Policy converged after %i iterations of value iteration" % i)
                break
            # Update value function with new estimate
            self.V = newV
                # ***************

    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        # **********
        # TODO 2.2
        return self.V[state]
        # **********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # ***********
        # TODO 2.3.
        probs = self.mdp.getTransitionStatesAndProbs(state,action)
        Q = 0.0 
        for prob in probs : 
            r = self.mdp.getReward(state,action,prob[0])
            Q += prob[1] * (r+self.discount * self.V[prob[0]])
        return Q
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """

        actions = self.mdp.getPossibleActions(state)
        if len(actions) < 1:
            return None

        else:

        # **********
        # TODO 2.4
            
            q_max = -np.inf
            a_max = None
            for action in actions:                
                q = self.getQValue(state, action)
                
                if q>q_max : 
                    q_max = q
                    a_max = action
            return a_max
        # ***********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for value iteration agents!
        """

        pass
