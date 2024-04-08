import tensorflow as tf
import gymnasium as gym
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Conv2D, Concatenate, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential # here we use functional API (not sequential)
import numpy as np
import random


## The core concept is to provide the task list and device list information as separate inputs 
## to the deep neural network and let the network learn how to combine them effectively for decision-making.

# Conceptual Architecture
# -----------------------
# Input Layers:
    # Task Input: An input layer designed to accept the feature vector representing the task list.
    # Device Input: Another input layer specifically designed for the feature vector representing the device list.

    # Input layer for task list: 
        # Shape:Have a 2D input layer to handle the all tasks observation with their features. So, the input layer 
        #       can have the shape (num_tasks, 8), accommodating all tasks with their 8 corresponding features.
        #       Here The DQN would process all tasks within the observation at once.


# Initial Processing:
    # Separate Dense Layers: Pass each input through its own set of dense (fully connected) layers. 
    # This allows the network to process the task and device information independently initially.

# Concatenation:
    # Combine Processed Information: After some initial processing, concatenate the outputs 
    # from the task-processing layers and device-processing layers. This merges the learned representations of tasks and devices.

# Combined Decision Layers:
    # Additional Dense Layers: Add more dense layers on top of the concatenated output. 
    # These layers will learn to interpret the combined task and device information.

# Output Layers:
    # One Output Layer:
    # The output layer would contain similar number of neurons to the number of maximum tasks, 
    # each corresponding to the Q-value for a specific task. The agent would select the task with the highest Q-value.


# import my environement
from WarehouseEnv_V0 import WarehouseEnv

#task_list_columns
TASK_ID = 0
TASK_TYPE = 1
TASK_PRODUCT = 2
TASK_FROM_LOC = 3
TASK_TO_LOC = 4
TASK_TIME = 5
TASK_ORDER = 6
TASK_STATUS = 7

# Helper function to choose an action based on the current state
def choose_random_action(tasks):
    # Choose an random task (action). But this task should be an available task
    available_tasks = [x for x in tasks if x[TASK_STATUS] == 0]  # Filter tasks with status 0 (available)
    available_task_ids = [sublist[0] for sublist in available_tasks]
    random_action = random.choice(available_task_ids) - 1 # Randomly select an available task. reduce 1 to match the index (if not 0 will never selected)
    print("Random action selected:", random_action)
    return random_action


# Helper class for the replay buffer 
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def store(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity: # if buffer list is less than capacity, we need to append None 
            # because we are going to use zip(*batch). This is needed to convert the batch into a format 
            # that can be used for training the neural network.
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch)) 


class TensorFlowSizeLayer(layers.Layer):
    def call(self, inputs):
      return tf.size(inputs, out_type=tf.int32)


# Create the DQN model
def create_dqn(tasks_shape, devices_shape, num_actions):
    print('\n==>[creating DQN model]: tasks_shape:', tasks_shape, 'devices_shape:', devices_shape, 'num_actions:', num_actions, '\n')

    # ------------------
    # Define Input Shapes (Adjust based on your exact features)
    # ------------------
    task_input_shape = tasks_shape  # 2D tensor (num_tasks, 8). 8 means number of features of each task
    device_input_shape = devices_shape  # 2D tensor (num_devices, 3). 3 means number of features of each device

    # make sure task_input_shape and device_input_shape are compatible to be concatenated
    if device_input_shape[0] < task_input_shape[0]:
        print('\n==>[creating DQN model]: Number of tasks and devices should match for concatenation. So increasing number of devices to match number of tasks\n')
        device_input_shape = (task_input_shape[0], device_input_shape[1])
        print('==>[creating DQN model]: After increasing, tasks_shape:', task_input_shape, 'devices_shape:', device_input_shape, 'num_actions:', num_actions, '\n')

    

    # ------------------
    # Define Input Layers
    # ------------------
    task_input = Input(shape=task_input_shape) # 
    device_input = Input(shape=device_input_shape) 

    # ------------------
    # Initial Task Processing 
    # ------------------
    task_x = Dense(32, activation='relu')(task_input)  # Adjust layers as needed
    task_x = Dense(16, activation='relu')(task_x)

    # ------------------
    # Initial Device Processing
    # ------------------
    device_x = Dense(16, activation='relu')(device_input)  # Adjust layers as needed

    # ------------------
    # Combine processed Information
    # ------------------
    combined = Concatenate()([task_x, device_x])

    # ------------------
    # Combined Decision Layers
    # ------------------ 
    x = Dense(32, activation='relu')(combined) 

    # ------------------
    # One layer after the combined decision layers
    # ------------------
    x1 = Dense(16, activation='relu')(x)

    # ------------------
    # Flatten the combined input before the output layer
    # ------------------
    flattened_x1 = Reshape((1, 160))(x1)  # Flatten the input for the output layer

    # ------------------
    # Output Layers (Adjust to match your number of tasks)
    # ------------------
    task_output = Dense(num_actions, activation='linear')(flattened_x1)  # Output for task selection (number of neurons = number of tasks)

    # ------------------
    # Instantiate the Model
    # ------------------
    model = Model(inputs=[task_input, device_input], outputs=[task_output])

    # ------------------
    # Compile the Model 
    # ------------------
    model.compile(optimizer='adam', loss='mse')  # You'll likely use Q-value loss functions in practice

    return model

# Training setup
# env = gym.make("CartPole-v1")  # Create the environment
env = WarehouseEnv()  # Create the environment
dqn = create_dqn(env.observation_space['tasks'].shape, env.observation_space['devices'].shape, env.action_space.n)
target_dqn = create_dqn(env.observation_space['tasks'].shape, env.observation_space['devices'].shape, env.action_space.n)
target_dqn.set_weights(dqn.get_weights())  # Initialize target network with same weights

buffer = ReplayBuffer(50000) # Initialize replay buffer with capacity of 50,000 experience samples
optimizer = Adam()
gamma = 0.99  # Discount factor for future rewards
batch_size = 10

# Training loop
for episode in range(1):
    (state, info) = env.reset()
    done = False
    total_reward = 0

    while not done:
        ### Following set of lines is for collecting training samples (i.e filling the experiance replay bugffer)
        action = choose_random_action(state['tasks'])  # Implement your action selection strategy
        next_state, reward, done, truncated, info = env.step(action)
        buffer.store(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        ## End of experiance replay ###
        print("\n ===> finished experiance replay")

        if len(buffer.buffer) > batch_size: # Start training once buffer has enough samples
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            # print('\n====>[Training]: states:', states, 'actions:', actions, 'rewards:', rewards, 'next_states:', next_states, 'dones:', dones, '\n')
            print("\nlen(states):", len(states), ",len(actions):", len(actions), ",len(rewards):", len(rewards), ",len(next_states):", len(next_states), ",len(dones):", len(dones))
            print("actions[0]:", actions)
            
            # get a batch of tasks and devices
            tasks = [[sublist for sublist in element['tasks']] for element in states]
            devices = [[sublist for sublist in element['devices']] for element in states]

            # get a batch of next tasks and devices
            next_tasks = [[sublist for sublist in element['tasks']] for element in next_states]
            next_devices = [[sublist for sublist in element['devices']] for element in next_states]

            print("\n===> the first observation in the selected batch from experiances for training =====>")
            print("tasks[0]:", tasks[0])
            print("\ndevices[0]:", devices[0])

            # Convert to TensorFlow tensors
            tasks_tensor = tf.convert_to_tensor(tasks, dtype=tf.float32)  # Specify float32 for common use 
            devices_tensor = tf.convert_to_tensor(devices, dtype=tf.float32)

            next_tasks_tensor = tf.convert_to_tensor(next_tasks, dtype=tf.float32)
            next_devices_tensor = tf.convert_to_tensor(next_devices, dtype=tf.float32)

            # Calculate Q-values from the main network
            q_values_from_main_network = dqn([tasks_tensor,devices_tensor])  # Initialize targets with current Q-values (shape is: batch_size, 1, 10)
            q_values_from_main_network = tf.reshape(q_values_from_main_network, (batch_size, 10)) # Reshaping the EagerTensor
            print("\n===> Q values from main network by passing a batch of experiances: =====>\n")
            print(q_values_from_main_network[0])
            print(q_values_from_main_network[1][7])

            # Get Q-values from target network
            q_values_from_target_network = target_dqn([next_tasks_tensor, next_devices_tensor])
            q_values_from_target_network = tf.reshape(q_values_from_target_network, (batch_size, 10)) # Reshaping the EagerTensor
            print("\n===> Q values from target network by passing a batch of experiances: =====>\n")
            print(q_values_from_target_network[1])

            # Calculate the maximum Q-values of the next states (from the target network)
            max_q_values_of_next_states = np.max(q_values_from_target_network, axis=1)
            print("\n===> Max Q values from target network by passing a batch of experiances: =====>\n")
            print(max_q_values_of_next_states)

            # Calculate the expected future rewards
            real_expected_future_rewards = q_values_from_main_network.numpy()  # Initialize with current Q-values from main network and convert to numpy array
            real_expected_future_rewards[range(batch_size), actions] = rewards + (1 - dones) * gamma * max_q_values_of_next_states
            print("\n===> Q values from main network, after updating the taken action with real expected future reward: =====>\n")
            print(real_expected_future_rewards)

            # Train the DQN
            with tf.GradientTape() as tape:
                q_values = dqn([tasks_tensor, devices_tensor])
                loss = tf.keras.losses.mean_squared_error(real_expected_future_rewards, q_values)
            grads = tape.gradient(loss, dqn.trainable_variables)
            optimizer.apply_gradients(zip(grads, dqn.trainable_variables))

    # Periodically update the target network
    if episode % 10 == 0:
        target_dqn.set_weights(dqn.get_weights()) 
