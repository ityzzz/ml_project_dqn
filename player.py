from game_snake import SnakeGame, Player
import numpy as np
import tkinter as tk
import time

MULTIPLIER = 10

class PlayerDrawable(Player):
    def __init__(self, canvas):
        self.bodyObj = []
        self.canvas = canvas

        super().__init__()

    def reset(self):
        self.bodyObj = []
        super().reset()
    
    def moveUpdate(self, foodPos, forceGrow=False):
        eat = super().moveUpdate(foodPos, forceGrow=forceGrow)
        while len(self.bodies) > len(self.bodyObj):
            self.bodyObj.append(canvas.create_rectangle(0, 0, MULTIPLIER, MULTIPLIER, fill="white"))
        
        for i in range(len(self.bodies)):
            b = self.bodies[i]
            v = self.bodyObj[i]
            self.canvas.coords(v, b[0] * MULTIPLIER, b[1] * MULTIPLIER, (b[0] + 1) * MULTIPLIER, (b[1] + 1) * MULTIPLIER)

        return eat




class SnakeGameDrawable(SnakeGame):
    def __init__(self, canvas):
        self.canvas = canvas
        self.foodObj = None
        super().__init__()

    def reset(self):
        self.foodObj = None
        self.canvas.delete("all")
        super().reset()

    def createPlayer(self):
        return PlayerDrawable(self.canvas)
    
    def createFood(self):
        super().createFood()

        if self.foodObj:
            self.canvas.coords(self.foodObj, self.foodPos[0] * MULTIPLIER, self.foodPos[1] * MULTIPLIER, (self.foodPos[0] + 1) * MULTIPLIER, (self.foodPos[1] + 1) * MULTIPLIER)
        else:
            self.foodObj = canvas.create_rectangle(self.foodPos[0] * MULTIPLIER, self.foodPos[1] * MULTIPLIER, (self.foodPos[0] + 1) * MULTIPLIER, (self.foodPos[1] + 1) * MULTIPLIER, fill="red")


window = tk.Tk()

window.title("SnakeGame")
window.geometry(str(SnakeGame.SPACE_WIDTH * MULTIPLIER) + "x" + str(SnakeGame.SPACE_HEIGHT * MULTIPLIER) + "+100+100")

canvas = tk.Canvas(window, width=SnakeGame.SPACE_WIDTH * MULTIPLIER, height=SnakeGame.SPACE_HEIGHT * MULTIPLIER, bg="black")
canvas.pack()

game = SnakeGameDrawable(canvas)

nextAction = 0
def leftKey(event):
    game.action(1)

def rightKey(event):
    game.action(2)

def reset(event):
    game.reset()

window.bind('<Left>', leftKey)
window.bind('<Right>', rightKey)
window.bind('x', reset)

while True:
    state, reward, dead = game.step(0)
    print(reward)

    canvas.update()
    window.update()
    time.sleep(0.2)