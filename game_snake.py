import numpy as np
import random
import time
from copy import deepcopy
from collections import deque

DEFAULT_SNAKE_LENGTH = 4

class SnakeGame:
	SPACE_WIDTH = 30
	SPACE_HEIGHT = 30

	def __init__(self):
		self.player = self.createPlayer()
		self.reset()
	
	def update(self):
		eat = self.player.moveUpdate(self.foodPos)

		if eat:
			self.createFood()
		
		return eat

	def createPlayer(self):
		return Player()

	def createFood(self):
		while True:
			rx = np.random.randint(self.SPACE_WIDTH)
			ry = np.random.randint(self.SPACE_HEIGHT)
			
			collision = False
			for p in self.player.bodies:
				if p[0] == rx and p[1] == ry:
					collision = True
					break
			
			if not collision:
				self.foodPos = [rx, ry]
				break

    
	def getState(self):
		state = np.zeros((self.SPACE_WIDTH, self.SPACE_HEIGHT))

		for p in self.player.bodies:
			state[p[0], p[1]] = 1
		
		head = self.player.getHeadPos()
		state[head[0], head[1]] = 2

		if self.foodPos:
			state[self.foodPos[0], self.foodPos[1]] = 3
		
		return state
	
	def reset(self):
		self.reward = 0
		self.player.reset()
		self.createFood()

		return self.getState()

	def action(self, i):
		if i == 1:
			self.player.turnLeft()
		elif i == 2:
			self.player.turnRight()

	def step(self, action):
		self.action(action)
		eat = self.update()

		if self.player.dead:
			reward = -1
		else:
			if eat:
				reward = len(self.player.bodies)
			else:
				reward = 0
		
		return self.getState(), reward, self.player.dead

class Player:
	LEFT = 0
	UP = 1
	RIGHT = 2
	DOWN = 3

	DEATH_REASON_COLLIDE_WALL = 1
	DEATH_REASON_COLLIDE_BODY = 2

	def __init__(self):
		self.reset()
	
	def reset(self):
		self.dead = False
		self.deadReason = 0
		self.bodies = deque()
		self.bodies.append([int(SnakeGame.SPACE_WIDTH / 2), int(SnakeGame.SPACE_HEIGHT / 2)])

		self.direction = np.random.randint(4)
		self.moveUpdate(None, forceGrow=True)
		self.moveUpdate(None, forceGrow=True)
		self.moveUpdate(None, forceGrow=True)

	def getLength(self):
		return len(self.bodies)
	
	def getHeadPos(self):
		return self.bodies[self.getLength() - 1]

	def moveUpdate(self, foodPos, forceGrow=False):
		if self.dead:
			return False

		nextPos = deepcopy(self.getHeadPos())
		if self.direction == self.LEFT:
			nextPos[0] -= 1

		elif self.direction == self.UP:
			nextPos[1] -= 1

		elif self.direction == self.RIGHT:
			nextPos[0] += 1

		elif self.direction == self.DOWN:
			nextPos[1] += 1

		if nextPos[0] < 0 or nextPos[0] >= SnakeGame.SPACE_WIDTH or nextPos[1] < 0 or nextPos[1] >= SnakeGame.SPACE_HEIGHT:
			self.dead = True
			self.deadReason = self.DEATH_REASON_COLLIDE_WALL
			return False
		
		for i in range(len(self.bodies)):
			# except tail
			if i == 0:
				continue
			
			if np.array_equal(self.bodies[i], nextPos):
				self.dead = True
				self.deadReason = self.DEATH_REASON_COLLIDE_BODY
				return False

		self.bodies.append(nextPos)

		eat = False
		if foodPos:
			eat = np.array_equal(foodPos, nextPos)
		
		if forceGrow:
			eat = True
		
		if not eat:
			self.bodies.popleft()
				
		return eat

	def turnLeft(self):
		self.direction = (self.direction - 1) % 4
	
	def turnRight(self):
		self.direction = (self.direction + 1) % 4
		