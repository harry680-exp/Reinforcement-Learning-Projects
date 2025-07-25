import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import dependencies
import random
import gymnasium as gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Environment setup
env = gym.make('CartPole-v1', render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
no_episodes = 1001
output_dir = './cartpole'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# DQN Agent
class DQNAgent:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.learning_rate = 0.001  # Fixed variable name
        
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def remember(self, current_state, action, reward, next_state, done):
        self.memory.append((current_state, action, reward, next_state, done))
        
    def act(self, state):    
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for current_state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            
            target_f = self.model.predict(current_state, verbose=0)  
            target_f[0][action] = target
            
            self.model.fit(current_state, target_f, epochs=1, verbose=0)  
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)

# Training
agent = DQNAgent(state_size, action_size)

for episode in range(no_episodes):
    state, _ = env.reset()
    state = np.reshape(state, (1, state_size))
    total_reward = 0

    for time in range(5000):
        env.render()
        
        action = agent.act(state)
        
        next_state, reward, done, truncated, info = env.step(action)  # Fixed: handle all return values
        
        if not done:
            reward = reward
        else:
            reward = -10
        
        next_state = np.reshape(next_state, (1, state_size))
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            print(f"Episode: {episode}/{no_episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
            break
        
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    
    # Save model periodically
    if episode % 100 == 0:
        agent.save(f"{output_dir}/dqn-{episode}.weights.h5")

# Save final model
agent.save(f"{output_dir}/dqn-final.weightsh5")
env.close()