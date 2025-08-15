import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from learn import Replaybuffer
from network import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, input_dims, max_size=100000, env=None, n_actions=2):
        self.gamma = 0.99
        self.tau = 0.0005
        self.n_actions = n_actions
        self.input_dims = input_dims
        
        self.memory = Replaybuffer(max_size, input_dims, self.n_actions)
        self.batch_size = 64
        self.noise = 0.3
        self.max_action = np.array(env.action_space.high).flatten()[0]
        self.min_action = np.array(env.action_space.low).flatten()[0]
        self.learning_rate_actor = 1e-5
        self.learning_rate_critic = 1e-4
        
         # Get actual state size from environment
        test_state = env.reset()
        if isinstance(test_state, tuple):
            test_state = test_state[0]
        
        actual_input_dims = [len(test_state)]  # ✅ Use actual size
        
        self.actor = ActorNetwork(n_actions=self.n_actions, name='actor')
        self.critic = CriticNetwork(n_actions=self.n_actions, name='critic')
        self.target_actor = ActorNetwork(n_actions=self.n_actions, name='target_actor')
        self.target_critic = CriticNetwork(n_actions=self.n_actions, name='target_critic')
        
        #--------
        actor_optimizer = Adam(learning_rate =self.learning_rate_actor,clipnorm=1.0)
        critic_optimizer = Adam(learning_rate=self.learning_rate_critic,clipnorm=1.0)
        
    
        
        
        self.actor.compile(optimizer=actor_optimizer)
        self.critic.compile(optimizer=critic_optimizer)
        self.target_actor.compile(optimizer=Adam(learning_rate=self.learning_rate_actor))
        self.target_critic.compile(optimizer=Adam(learning_rate=self.learning_rate_critic))
        
        self.update_network_parameter(tau=1)
        
        # At the end of __init__, after self.update_network_parameter(tau=1)
        dummy_state = tf.zeros((1, *actual_input_dims))
        dummy_action = tf.zeros((1, self.n_actions))
        _ = self.actor(dummy_state)
        _ = self.target_actor(dummy_state)
        _ = self.critic(dummy_state, dummy_action)
        _ = self.target_critic(dummy_state, dummy_action)
        
    def update_network_parameter(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        new_weights = [tau * aw + (1 - tau) * tw for aw, tw in zip(actor_weights, target_actor_weights)]
        self.target_actor.set_weights(new_weights)

        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        new_critic_weights = [tau * cw + (1 - tau) * tcw for cw, tcw in zip(critic_weights, target_critic_weights)]
        self.target_critic.set_weights(new_critic_weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def save_models(self):
        print('Saving models...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)
    
    def load_models(self):
        print('Loading models...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)
        
    def action(self, observation, evaluate=False):
        # Handle tuple (e.g., observation, info)
        if isinstance(observation, tuple):
            observation = observation[0]

        # Convert to NumPy array safely
        observation = np.array(observation, dtype=np.float32)

        # Add batch dimension
        state = tf.convert_to_tensor(observation[np.newaxis, :], dtype=tf.float32)

        # Get raw action from actor network
        actions = self.actor(state)

        # Add exploration noise (if training)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)

        # Clip to valid action bounds
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0].numpy().flatten()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        dones = tf.convert_to_tensor(done, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(state, actions), 1)
            target = rewards + self.gamma * critic_value_ * (1 - dones)
            critic_loss = keras.losses.MSE(target, critic_value)
        
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(state)
            actor_loss = -self.critic(state, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
        
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        
        self.update_network_parameter()
