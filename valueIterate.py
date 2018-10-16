import numpy as np
import pdb
import random
from copy import deepcopy
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class robot:
    #init robot (errorPr, discount)
    #errorPr: the probability of pre-rotation error
    #discount: the discount factor, gamma, for time weighted reward
    def __init__(self, errorPr=0,discount = 1):
        #problem 1a
        self.valueMatrix = np.array([[[0.0 for _ in range(12)] for _ in range(6)] for _ in range(6)])
        #Problem 1b
        # action Space: 'F' for forward, 'B' for backward, '0' for still,
        #               '+' for clockwise, '-' anticlockwise
        self.actionSpace = [['F', '+'], ['F', '0'], ['F', '-'],
                            ['B', '+'], ['B', '0'], ['B', '-'],
                            ['0', '0']]
        #Policy matrix
        self.actionMatrix = [[[['0', '0'] for _ in range(12)] for _ in range(6)] for _ in range(6)]
        self.discount = discount
        self.errorPr = errorPr
        #Reward map
        self.reward = [[-100, -100, -100, -100, -100, -100],
                       [-100, 0,    0,    0,    0,    -100],
                       [-100, 0,    -10,  0,    -10,  -100],
                       [-100, 0,    -10,  0,    -10,  -100],
                       [-100, 0,    -10,  1,    -10,  -100],
                       [-100, -100, -100, -100, -100, -100],]




    #Problem 1c
    # Psas' fucntion (currentState,nextState,action)
    # return the probability of next state given action and current state
    #
    #currentState: indicate the current state of robot [y,x,rotation]
    #nextState: indicate the next state of robot [y,x,rotation]
    #action: an action from Action Space
    def probActionState(self, currentState, nextState, action):
        if action[0] == '0':
            return 1 if currentState == nextState else 0

        if abs(currentState[0]-nextState[0])>1 or abs(currentState[1]-nextState[1])>1:
            return 0
        else:
            intentNextState = self.computeNextState(currentState, action, False)
            temp = currentState[:]
            temp[2] = (temp[2]+1)%12
            errorNextState1 = self.computeNextState(temp, action, False)
            temp = currentState[:]
            temp[2] = (temp[2]-1)%12
            errorNextState2 = self.computeNextState(temp, action, False)

            if intentNextState == nextState:
                return 1-2*self.errorPr
            elif nextState == errorNextState1 or nextState == errorNextState2:
                return self.errorPr
            else:
                return 0

    #Problem 1d
    #computeNextState (currentState,action,prerotateError)
    #return next state give current action and current state
    #
    #currentState: indiciate the current state of robot [y,x,rotation]
    #actoin: an action from Action Space
    #prerotateError: a boollean to indicate if pre-rotation error is in consideration
    def computeNextState(self, currentState, action, prerotateError=True):
        if action[0] == '0':
            return currentState

        if prerotateError:
            temp = random.uniform(0, 1)
            if temp < self.errorPr:
                preError = '-'
                currentState[2] = (currentState[2]-1)%12
            elif temp < 2*self.errorPr:
                preError = '+'
                currentState[2] = (currentState[2]+1)%12
            else:
                preError = '0'

        up, down, right, left = set([11,0,1]), set([7,6,5]), set([2,3,4]), set([8,9,10])

        nextState = currentState[:]
        if currentState[2] in up:
            nextState[0] = currentState[0]+1 if action[0] == 'F' else currentState[0]-1
        elif currentState[2] in down:
            nextState[0] = currentState[0]-1 if action[0] == 'F' else currentState[0]+1
        elif currentState[2] in right:
            nextState[1] = currentState[1]+1 if action[0] == 'F' else currentState[1]-1
        else:
            nextState[1] = currentState[1]-1 if action[0] == 'F' else currentState[1]+1

        if action[1] == '+':
            nextState[2] = (nextState[2]+1)%12
        elif action[1] == '-':
            nextState[2] = (nextState[2]-1)%12

        for i in range(2):
            if nextState[i] < 0:
                nextState[i] = 0
            elif nextState[i] > 5:
                nextState[i] = 5

        return nextState

    #Problem 2a
    # getReward(state)
    # return the reward points given state
    #
    # state: robot state [y,x,rotation]
    def getReward(self, state):
        return self.reward[state[0], state[1]]

    #Problem 4a
    # valueIteration(horizon)
    # this function perform value iteration
    # horizon: limit the maximum iteration, in case of convergence never happened
    #
    # return valueMatrix, actionMatrix
    # valueMatrix: the value of each state
    # actionMatrix: policy matrix, each state corresponding to one action
    def valueIteration(self, horizon):
        self.valueMatrix = np.zeros((6,6,12)) #reset valueMatrix to zeros
        for n in range(horizon): #iterate until meet horizon

            #assign update value to Q(s',a)
            valueHolder = deepcopy(self.valueMatrix)

            #Iterate through all Current State
            for i in range(6):
                for j in range(6):
                    for k in range(12):

                        actionValueCollection = [] #Hold 7 Q(s,a)

                        #iterate through 7 action
                        for a in range(7):
                            currentState = [i,j,k]
                            Qsa = 0.0
                            nextState = []
                            #possible next three state given action
                            nextState.append(self.computeNextState([i, j, k], self.actionSpace[a], False))
                            nextState.append(self.computeNextState([i, j, (k+1)%12], self.actionSpace[a], False))
                            nextState.append(self.computeNextState([i, j, (k-1)%12], self.actionSpace[a], False))

                            #Calculate Q(s,a)
                            for state in nextState:
                                Qsa += self.probActionState(currentState,state,self.actionSpace[a]) * \
                                        (float(self.reward[i][j])+ self.discount * valueHolder[state[0]][state[1]][state[2]])
                            #Store Q(s,a)
                            actionValueCollection.append(Qsa)
                        Qsa = 0.0 #reset Qsa value

                        #Compare Q(s,a1)~Q(s,a7), choose the action with highest Q(s,a)
                        self.valueMatrix[i][j][k] = np.max(actionValueCollection)
                        self.actionMatrix[i][j][k] = self.actionSpace[actionValueCollection.index(np.max(actionValueCollection))]

            #exist loop early if converge
            if np.all(self.valueMatrix.round(0) == valueHolder.round(0)):
                print( 'number of iteration {}'.format(n))
                break

        return self.valueMatrix.round(0), self.actionMatrix


    def getTrajectory(self, startPoint, actionMatrix):
        result = []
        idx = startPoint[:]
        result.append(idx)
        i = 0
        print(idx)
        while idx[0] != 4 or idx[1] != 3:
            idx = self.computeNextState(idx, actionMatrix[idx[0]][idx[1]][idx[2]],prerotateError=False)
            if idx in result:
                print(idx)
                break
            result.append(idx)
            print(idx)
        return result


    def plotTrajetory(self, trajectory):
        value = [self.valueMatrix[tra[0]][tra[1]][tra[2]] for tra in trajectory]
        print('value of the trajectory is: {}'.format(value))
        print('sum of the value of the trajectory: {:.3f}'.format(sum(value)))
        tra = trajectory

        plt.plot([t[1] for t in tra], [t[0] for t in tra])
        plt.ylabel('y-axis')
        plt.xlabel('x-axis')
        plt.title('trajectory of an action')
        plt.ylim(0,5)
        plt.xlim(0,5)
        plt.grid()
        plt.show()

if __name__ == '__main__':
        #Problem 4b
        robot = robot(.0,0.9)  #initialize with error probability and discount factor
        #problem 5a
        #robot = robot(0.25,0.9)
        a = time.time()
        robot.valueIteration(1000) #horizon set to 1000
        b= time.time()
        print('Time comsume of value iteration: {:.3f}'.format(b-a))
        result = robot.getTrajectory([4,1,6], robot.actionMatrix)
        robot.plotTrajetory(result)
