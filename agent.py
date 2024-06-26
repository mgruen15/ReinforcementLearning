import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot 

# This codebase is adapted from https://github.com/patrickloeber/snake-ai-pytorch

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.005
STEP_PENALTY = -0.01

class Agent: 
    
    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0 # control randomness
        self.gamma = 0.9 # discount factor
        self.memory = deque(maxlen=MAX_MEMORY) # if we exceed memory, it will call popleft()
        self.model = Linear_QNet(11, 256, 3) # eleven points of information for each state, output is 3 because we have three possible actions
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0] # get the head

        point_l = Point(head.x - 20, head.y) # check if there is danger around the head
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int) # through the dtype=int we convert the True/False values into 1s and 0s


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # If the maximum memory is reached, we execute popleft()

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample) # more elegant than a for loop
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # begin with random moves -> exploration, exploitation trade-off
        self.epsilon = 80 - self.n_games # hyperparameter, which can be changed
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0, 2) # do a random move -> exploration
            final_move[move] = 1 # save random choice in move parameter
        else:
            state0 = torch.tensor(state, dtype=torch.float) # exploitation
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game) # get the current state
        final_move = agent.get_action(state_old) # get the move from agent
        reward, done, score = game.play_step(final_move) # perform move 
                
        reward += STEP_PENALTY # Apply step penalty

        state_new = agent.get_state(game) # get new state

        agent.train_short_memory(state_old, final_move, reward, state_new, done) # train short memory
        agent.remember(state_old, final_move, reward, state_new, done) # remember

        if done: 
            # train long memory (Experienced Replay), plot result 
            game.reset()

            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score # if a new record has been reached, update the record in the plot
                agent.model.save()
            print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores, agent.n_games, record)

if __name__ == '__main__':
    train()