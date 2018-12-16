import tensorflow as tf
import numpy as np
import random
import time
from collections import deque

from game_snake import SnakeGame
from dqn import DQN

MAX_EPISODES = 3000
REPLAY_MEMORY = 600
DISCOUNT_FACTOR = 0.95
RANDOM_PROB = 1e-3

BATCH_REPEAT_COUNT = 50
BATCH_SAMPLE_COUNT = 10

OBSERVE_EP = 30

SAVE_PERIOD = 100

MAXIMUM_TICK = 1500

ACTION_COUNT = 3
EXPLORE_PROB = .001

def replay_train(main, target, train_batch):
    x_stack = np.empty(0).reshape(0, SnakeGame.SPACE_WIDTH * SnakeGame.SPACE_HEIGHT)
    y_stack = np.empty(0).reshape(0, ACTION_COUNT)

    for state, action, reward, next_state, end in train_batch:
        Q = main.predict(state)

        if end:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + DISCOUNT_FACTOR * np.max(target.predict(next_state))
        
        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])

    return main.update(x_stack, y_stack)

def train():
    replay_buffer = deque()

    sess = tf.Session()

    mainDQN = DQN(sess, SnakeGame.SPACE_WIDTH, SnakeGame.SPACE_HEIGHT, ACTION_COUNT, "main")
    targetDQN = DQN(sess, SnakeGame.SPACE_WIDTH, SnakeGame.SPACE_HEIGHT, ACTION_COUNT, "target")
    sess.run(tf.global_variables_initializer())

    copy_ops = []
    src = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="main")
    dest = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")

    for src, dest in zip(src, dest):
        copy_ops.append(dest.assign(src.value()))

    sess.run(copy_ops)

    game = SnakeGame()
    
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('saved')
    if ckpt is not None:
        saver.restore(sess, ckpt.model_checkpoint_path)

    epsilon = 1.

    for i in range(MAX_EPISODES):
        terminated = False
        tick = 0

        total_reward = 0
        state = np.reshape(game.reset(), [1, SnakeGame.SPACE_WIDTH * SnakeGame.SPACE_HEIGHT])

        if i > OBSERVE_EP:
            epsilon -= .001

        while not terminated:
            if np.random.rand(1) < epsilon or np.random.rand(1) < EXPLORE_PROB:
                action = np.random.randint(ACTION_COUNT)
            else:
                action = np.argmax(mainDQN.predict(state))
            
            next_state, reward, terminated = game.step(action)
            next_state = np.reshape(next_state, [1, SnakeGame.SPACE_WIDTH * SnakeGame.SPACE_HEIGHT])

            total_reward += reward
            replay_buffer.append((state, action, reward, next_state, terminated))
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()
            
            state = next_state
            tick += 1
            if tick > MAXIMUM_TICK:
                break
        
        print("{}\t{}".format(i, total_reward))

        if i % 10 == 1:
            for _ in range(BATCH_REPEAT_COUNT):
                batch = random.sample(replay_buffer, BATCH_SAMPLE_COUNT)
                loss, _ = replay_train(mainDQN, targetDQN, batch)
            
            sess.run(copy_ops)
        
        if i % SAVE_PERIOD == 0:
            saver.save(sess, 'saved/dqn.ckpt', global_step=i)

    print('finish!')




def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()
