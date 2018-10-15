import numpy as np
import pdb
import random
import matplotlib
import time
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class robot:
	def __init__(self, errorPr=0, discount=1):
		self.errorPr = errorPr
		self.discount = discount
		self.actionSpace = [['F', '+'], ['F', '0'], ['F', '-'],
							['B', '+'], ['B', '0'], ['B', '-'],
							['0', '0']]
		self.probMatrix = np.array([[0.0]*432 for i in range(432)])
		#self.row, self.column = 6, 6
		self.reward = [[-100, -100, -100, -100, -100, -100],
					   [-100, 0,	0,	  0, 	0,	  -100],
					   [-100, 0,	-10,  0,	-10,  -100],
					   [-100, 0,	-10,  0,	-10,  -100],
					   [-100, 0,	-10,  1,	-10,  -100],
					   [-100, -100, -100, -100, -100, -100],
						]
		self.valueMatrix = np.array([[[0.0 for _ in range(12)] for _ in range(6)] for _ in range(6)])
		self.actionMatrix = [[[['0', '0'] for _ in range(12)] for _ in range(6)] for _ in range(6)]

	def computeNextState(self, currentState, action, prerotateError=True):
		# prerotateError manually control
		# currentState & nextState: [y, x, head]
		# action ['F', '+']: 'F' for forward, 'B' for backward, '0' for still, '+' for clockwise, '-' anticlockwise
		if action[0] == '0' and action[1] == '0': return currentState
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

	def computeNextStateList(self, currentState, action):
		result = []
		if action[0] == '0' and action[1] == '0':
			return [currentState]
		else:
			result.append(self.computeNextState(currentState, action, False))
			temp = currentState[:]
			temp[2] = (temp[2]-1)%12
			result.append(self.computeNextState(currentState, action, False))
			temp = currentState[:]
			temp[2] = (temp[2]+1)%12
			result.append(self.computeNextState(currentState, action, False))
			return result


	def probActionState(self, currentState, nextState, action):
		# currentState & nextState: [y, x, head]
		# action ('F', '+'): 'F' for forward, 'B' for backward, '0' for still, '+' for clockwise, '-' anticlockwise
		if action[0] == '0' and action[1] == '0':
			return 1 if nextState == currentState else 0

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

	def getConditionalReward(self, state):
		down = set([5,6,7])
		return 1.0 if state[0]==4 and state[1]==3 and state[2] in down else 0.0

	def initialPolicy(self):
		# flood the actionMatrix with initial policy
		# goal square idx (y, x) = (4, 3)
		def computeDistance(currentState, nextState):
			return sum([abs(currentState[i]-nextState[i]) for i in range(2)])

		up, down, right, left = set([11,0,1]), set([7,6,5]), set([2,3,4]), set([8,9,10])
		for y in range(6):
			for x in range(6):
				for h in range(12):
					currentAction = ['0', '0']
					#print(y,x,h)
					if y == 4 and x == 3:
						#pdb.set_trace()
						self.actionMatrix[y][x][h] = currentAction
						continue
					if h in up:
						currentAction[0] = 'F' if y < 4 else 'B'
					elif h in down:
						currentAction[0] = 'F' if y > 4 else 'B'
					elif h in right:
						currentAction[0] = 'F' if x < 3 else 'B'
					else:
						currentAction[0] = 'F' if x > 3 else 'B'
					#tempNextState = self.computeNextState([y, x, h], currentAction, prerotateError = False)
					action1, action2, action3 = [currentAction[0], '+'], [currentAction[0], '0'], [currentAction[0], '-']
					nextState1, nextState2, nextState3 = self.computeNextState([y, x, h], action1, prerotateError = False), \
														self.computeNextState([y, x, h], action2, prerotateError = False), \
														self.computeNextState([y, x, h], action3, prerotateError = False)
					actionList = [action1, action1, action2, action2, action3, action3]
					stateList = [self.computeNextState(nextState1, ['F', '0'], prerotateError=False),
								 self.computeNextState(nextState1, ['B', '0'], prerotateError=False),
								 self.computeNextState(nextState2, ['F', '0'], prerotateError=False),
								 self.computeNextState(nextState2, ['B', '0'], prerotateError=False),
								 self.computeNextState(nextState3, ['F', '0'], prerotateError=False),
								 self.computeNextState(nextState3, ['B', '0'], prerotateError=False),]

					List = [[computeDistance([4, 3], stateList[i][:2]), actionList[i]] for i in range(6)]
					List = sorted(List, key = lambda x:x[0])
					#if y == 0 and x == 0 and h == 3:
						#pdb.set_trace()
					self.actionMatrix[y][x][h] = List[0][1]

	def getTrajectory(self, startPoint, actionMatrix):
		result = []
		idx = startPoint[:]
		result.append(idx)
		i = 0
		while idx[0] != 4 or idx[1] != 3:
			idx = self.computeNextState(idx, actionMatrix[idx[0]][idx[1]][idx[2]], prerotateError=False)
			if idx in result: break
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

	def computeValue(self, iteration=20, modified=False):
		def isAjcent(currentState, nextState):
			return False if sum([abs(currentState[i]-nextState[i]) for i in range(2)]) > 1 \
					else True
		valueMatrix = self.valueMatrix[:]
		for _ in range(iteration):
			for i in range(6):
				for j in range(6):
					for k in range(12):
						currentState = [i, j, k]
						nextState = []
						action = self.actionMatrix[i][j][k]
						nextState.append(self.computeNextState([i, j, k], action, False))
						nextState.append(self.computeNextState([i, j, (k+1)%12], action, False))
						nextState.append(self.computeNextState([i, j, (k-1)%12], action, False))
						#if i == 4 and j == 0 and k == 0: pdb.set_trace()
						if not modified:
							self.valueMatrix[i][j][k] = sum([self.probActionState(currentState, state, action)*\
								   						valueMatrix[state[0]][state[1]][state[2]] \
								   						for state in nextState])*self.discount + self.reward[i][j]
						else:
							self.valueMatrix[i][j][k] = sum([self.probActionState(currentState, state, action)*\
								   						valueMatrix[state[0]][state[1]][state[2]] \
								   						for state in nextState])*self.discount + self.getConditionalReward(currentState)
	def computeValuePesudoInverse(self):
		def isAjcent(currentState, nextState):
			return False if sum([abs(currentState[i]-nextState[i]) for i in range(2)]) > 1 \
					else True
		#x = np.resize(valueMatrix, (432, 1))
		idx = [[[[i, j, k] for k in range(12)] for j in range(6)] for i in range(6)]
		idx = np.resize(idx, (432, 3))
		idxDict = dict(zip([i for i in range(432)], idx))
		A = np.array([[0.0]*432 for i in range(432)])
		for i in range(len(idx)):
			for j in range(len(idx)):
				currentState, nextState = [idxDict[i][k] for k in range(3)], [idxDict[j][k] for k in range(3)]
				if isAjcent(currentState, nextState):
					A[i][j] = self.probActionState(currentState, nextState, \
							self.actionMatrix[currentState[0]][currentState[1]][currentState[2]])*self.discount
				else:
					A[i][j] = 0.0
		self.probMatrix = A
		b = []
		for i in range(432):
			b.append(self.reward[idx[i][0]][idx[i][1]])

		x = np.dot(np.linalg.pinv(np.identity(432)-A), b)
		self.valueMatrix = np.resize(x, (6,6,12))


	def updatePolicy(self):
		updated = False
		for i in range(6):
			for j in range(6):
				for k in range(12):
					actionSpace = self.actionSpace[:]
					currentState = [i, j, k]
					pair = []
					#pdb.set_trace()
					for action in actionSpace:
						nextState = []
						nextState.append(self.computeNextState([i, j, k], action, False))
						nextState.append(self.computeNextState([i, j, (k+1)%12], action, False))
						nextState.append(self.computeNextState([i, j, (k-1)%12], action, False))

						value = ([self.probActionState(currentState, state, action)*\
							   self.valueMatrix[state[0]][state[1]][state[2]] for state in nextState])
						pair.append((sum(value), action))
						#if i == 2 and j == 1 and k == 9: pdb.set_trace()
					pair = sorted(pair, key=lambda x:x[0], reverse=True)
					if self.actionMatrix[i][j][k] != pair[0][1]:
						self.actionMatrix[i][j][k] = pair[0][1]
						updated = True
		return updated

	def policyIteration(self, iteration=20, modified=False):
		self.initialPolicy()
		updated = True

		while updated:
			self.computeValue(iteration, modified)
			updated = self.updatePolicy()



if __name__ == '__main__':
	robot = robot(errorPr=0.1, discount=0.9)
	modifiedReward = False
	#print(robot.computeNextState([0,0,1], ['B', '+']))
	#print(robot.probActionState([5,5,11],[5,5,1], ['F', '+']))
	robot.initialPolicy()
	#result = robot.getTrajectory([5,5,0], robot.actionMatrix)
	#robot.computeValue()
	robot.computeValue(iteration=20, modified=modifiedReward)
	#pdb.set_trace()
	robot.updatePolicy()
	#robot.policyIteration()
	a = time.time()
	robot.policyIteration(iteration=20, modified=modifiedReward)
	b = time.time()
	print('Time comsume of policy iteration: {:.3f}'.format(b-a))
	result = robot.getTrajectory([4,1,0], robot.actionMatrix)
	#robot.plotTrajetory(result)
	#pdb.set_trace()
	print(robot.actionMatrix)
