import numpy as np
import pdb
import random
from copy import deepcopy

class robot:
    def __init__(self, errorPr=0):
        self.errorPr = errorPr
    	self.reward = [[-100, -100, -100, -100, -100, -100],
    				   [-100, 0,	0,	  0, 	0,	  -100],
    				   [-100, 0,	-10,  0,	-10,  -100],
    				   [-100, 0,	-10,  0,	-10,  -100],
    				   [-100, 0,	-10,  1,	-10,  -100],
    				   [-100, -100, -100, -100, -100, -100],]
    	self.valueMatrix = np.array([[[0.0 for _ in range(12)] for _ in range(6)] for _ in range(6)])
    	self.actionMatrix = [[[('0', '0') for _ in range(12)] for _ in range(6)] for _ in range(6)]
        self.actionSpace = [('F','+'),('F','-'),('F','0'),('B','+'),('B','-'),('B','0'),('0','0')]



    def computeNextState(self, currentState, action, prerotateError=True):
    	# prerotateError manually control
    	# currentState & nextState: [y, x, head]
    	# action ('F', '+'): 'F' for forward, 'B' for backward, '0' for still, '+' for clockwise, '-' anticlockwise
        if action == ('0','0'):
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
        if action == ('0','0'):
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

    def valueIteration(self, horizon,discount):
        valueHolder = self.valueMatrix #a value matrix holder to hold update value in each nextState

        #populate valueHolder with intial value 0.0
        for i in range(6):
            for j in range(6):
                for k in range(12):
                    valueHolder[i][j][k] = 0.0

        new_valueHolder = deepcopy(valueHolder) #declare new_valueHolder for output

        for n in range(horizon): #iterate until meet horizon
            valueHolder = deepcopy(new_valueHolder) #assign update value to V(s')

            #for all state
            for i in range(6):
                for j in range(6):
                    for k in range(12):

                        actionValueCollection = [0] #clear a temp value to compare 7 action's value

                        #iterate through 7 action
                        for a in range(7):
                            currentState = [i,j,k]
                            temp = 0.0
                            #iterate through next state
                            for y in range(6):
                                for x in range(6):
                                    for z in range(12):
                                            # if(valueHolder[y][x][z] > 0.0):
                                        #     print(y,x,z)
                                        #     print(valueHolder[y][x][z])
                                        temp += self.probActionState(currentState,[y,x,z],self.actionSpace[a]) * (float(self.reward[i][j]) + discount * valueHolder[y][x][z])

                            actionValueCollection.append(temp)
                            temp = 0.0
                        actionValueCollection.pop(0) #remove place holder, first component

                        new_valueHolder[i][j][k] = np.max(actionValueCollection)

        #print(valueHolder)
        #print (new_valueHolder)
        return new_valueHolder

        # def possibleNextState(self,currentState):
        #     temp = currentState
        #     if (currentState[0]+1 <= 5):
        #         temp.append [currentState[0]+1,currentState[1],currentState[2]]
        #         temp.append [currentState[0]+1,currentState[1],currentState[2]-1]
        #         temp.append [currentState[0]+1,currentState[1],currentState[2]+1]
        #         temp.append [currentState[0]+1,currentState[1],currentState[2]+2]
        #         temp.append [currentState[0]+1,currentState[1],currentState[2]-2]
        #     temp = currentState[]
        #     currentState[0]

	#def initialPolicy(self):


if __name__ == '__main__':
        robot = robot(0.25)
        print(robot.computeNextState([0,0,3], ('F', '+')))
        print(robot.probActionState([0,0,3],[0,1,4], ('F', '+')))
        #print(robot.probActionState([5,5,11],[5,5,11],('0','0')))
        print(robot.valueIteration(3,0.9))
        #print(robot.reward[1][0])
