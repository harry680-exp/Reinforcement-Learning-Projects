import numpy as np

class Replaybuffer:
    def __init__(self, max_size, input_shape, n_action):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_action))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size

        # Handle tuple observations (observation, info)
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(new_state, tuple):
            new_state = new_state[0]

        self.state_memory[index] = np.array(state, dtype=np.float32)
        self.new_state_memory[index] = np.array(new_state, dtype=np.float32)
        self.action_memory[index] = np.array(action, dtype=np.float32)
        self.reward_memory[index] = np.array(reward, dtype=np.float32)
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch] 
        next_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, next_states, dones  # Fixed: added return statement
