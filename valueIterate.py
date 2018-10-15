import numpy as np
import pdb
import random
from copy import deepcopy
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class robot:
    def __init__(self, errorPr=0):
        self.errorPr = errorPr
        self.reward = [[-100, -100, -100, -100, -100, -100],
                       [-100, 0,    0,    0,    0,    -100],
                       [-100, 0,    -10,  0,    -10,  -100],
                       [-100, 0,    -10,  0,    -10,  -100],
                       [-100, 0,    -10,  1,    -10,  -100],
                       [-100, -100, -100, -100, -100, -100],]
        self.valueMatrix = np.array([[[0.0 for _ in range(12)] for _ in range(6)] for _ in range(6)])
        self.actionMatrix = [[[['0', '0'] for _ in range(12)] for _ in range(6)] for _ in range(6)]
        self.actionSpace = [['F', '+'], ['F', '0'], ['F', '-'],
                            ['B', '+'], ['B', '0'], ['B', '-'],
                            ['0', '0']]



    def computeNextState(self, currentState, action, prerotateError=True):
        # prerotateError manually control
        # currentState & nextState: [y, x, head]
        # action ('F', '+'): 'F' for forward, 'B' for backward, '0' for still, '+' for clockwise, '-' anticlockwise
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

    def probActionState(self, currentState, nextState, action):
        # currentState & nextState: [y, x, head]
        # action ('F', '+'): 'F' for forward, 'B' for backward, '0' for still, '+' for clockwise, '-' anticlockwise
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

    def getReward(self, state):
        return self.reward[state[0], state[1]]

# A Value iteration function: (horizon, discount)
# horizon: limit the maximum iteration, in case of convergence never happened
# discount: discount factor, for reward point
# return valueMatrix, actionMatrix
    def valueIteration(self, horizon,discount):
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
                                        (float(self.reward[i][j])+ discount * valueHolder[state[0]][state[1]][state[2]])
                            #Store Q(s,a)
                            actionValueCollection.append(Qsa)
                        Qsa = 0.0

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
        while idx[0] != 4 or idx[1] != 3:
            idx = self.computeNextState(idx, actionMatrix[idx[0]][idx[1]][idx[2]],prerotateError=False)
            if idx in result:
                print(idx)
                break
            result.append(idx)
            print(idx)
        return result


    def plotTrajetory(self, trajectory):
        tra = trajectory
        plt.plot([t[1] for t in tra], [t[0] for t in tra])
        plt.ylabel('y-axis')
        plt.xlabel('x-axis')
        plt.title('trajectory of an action')
        plt.grid()
        plt.show()

if __name__ == '__main__':
        robot = robot(0)
        a = time.time()
        robot.valueIteration(1000,0.9)
        b= time.time()
        print('Time comsume of value iteration: {:.3f}'.format(b-a))
        result = robot.getTrajectory([4,1,6], robot.actionMatrix)
        robot.plotTrajetory(result)
