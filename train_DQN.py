import tensorflow as tf
import gymnasium as gym
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Conv2D, Concatenate, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential # here we use functional API (not sequential)
import numpy as np
import pandas as pd
import random
import keras
from keras import ops



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

# agent_columns
AGENT_ID = 0
AGENT_TYPE = 1

# agent type encodings
HUMAN = 0
ROBOT = 1

# agent status encodings
AVAILABLE = 1
ACTIVE = -1

# Helper function to choose an action based on the current state
# Helper function to choose an action based on the current state
def choose_random_action(tasks):
    # Choose an random task (action). But this task should be an available task
    available_tasks = [x for x in tasks if x[env.TASK_STATUS] == env.AVAILABLE]  # Filter tasks with status 0 (available)
    available_task_ids = [sublist[0] for sublist in available_tasks]

    if len(available_task_ids) > 0:
        random_action = random.choice(available_task_ids) # Randomly select an available task. (task and device ids start from 1)
        print(f"Random action selected:", random_action)
        return random_action
    else:
        print("No available tasks to choose from. Returning None.")
        return None


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
    task_input_shape = tasks_shape  # 2D tensor (num_tasks, 9). 9 means number of features of each task
    # device_input_shape = devices_shape  # 2D tensor (num_devices, 4). 4 means number of features of each device

    # ------------------
    # Define Input Layers
    # ------------------
    task_input = Input(shape=task_input_shape) # 
    # device_input = Input(shape=device_input_shape) 

    # ------------------
    # Initial Task Processing 
    # ------------------
    task_x = Dense(10, activation='relu')(task_input)  # Adjust layers as needed
    # task_x = Dense(100, activation='relu')(task_x)

    # ------------------
    # Initial Device Processing
    # ------------------
    # device_x = Dense(16, activation='relu')(device_input)  # Adjust layers as needed

    # ------------------
    # Combine processed Information
    # ------------------
    # combined = Concatenate()([task_x, device_x])

    # ------------------
    # Combined Decision Layers
    # ------------------ 
    # x = Dense(32, activation='relu')(combined) 

    # ------------------
    # One layer after the combined decision layers
    # ------------------
    # x1 = Dense(100, activation='relu')(task_x)  # Adjust layers as needed

    # ------------------
    # Flatten the combined input before the output layer
    # ------------------
    flattened_x1 = Reshape((1, 10 * num_actions))(task_x)  # Flatten the input for the output layer

    # flattened_x1 = Dense(num_actions, activation='relu')(flattened_x1)  # Flatten the input for the output layer

    # ------------------
    # Output Layers (Adjust to match your number of tasks)
    # ------------------
    task_output = Dense(num_actions, activation='linear')(flattened_x1)  # Output for task selection (number of neurons = number of tasks)

    # ------------------
    # Instantiate the Model
    # ------------------
    model = Model(inputs=[task_input], outputs=[task_output])

    # ------------------
    # Compile the Model 
    # ------------------
    # You'll likely use Q-value loss functions in practice
    model.compile(optimizer=keras.optimizers.Adam(), 
                  loss='mean_squared_error',
                  metrics=['accuracy', 'mean_squared_error'])  

    return model

  # Training setup
# env = gym.make("CartPole-v1")  # Create the environment
env = WarehouseEnv()  # Create the environment
dqn = create_dqn((60,3), env.observation_space['devices'].shape, env.action_space.n) #  None for variable number of observations
target_dqn = create_dqn((60,3), env.observation_space['devices'].shape, env.action_space.n)
target_dqn.set_weights(dqn.get_weights())  # Initialize target network with same weights

buffer = ReplayBuffer(50000) # Initialize replay buffer with capacity of 50,000 experience samples
optimizer = Adam()
gamma = 0.99  # Discount factor for future rewards
batch_size = 10


# Training loop
training_agent_id = 2
losses = []

for episode in range(11):
    done = False
    total_reward = 0
    time_step = -1

    # Reset the environment
    env.reset()

    while not done:
        time_step += 1
        
        env.time_step = time_step #update env time step

        for index, agent in env.agents.iterrows():

            agent_id = agent['agent_id']
            current_action = agent["current_task"]
            agent_device = agent['current_device']
            agent_device_index = agent_device - 1
            agent_type = agent['type']

            ### Following set of lines is for collecting training samples (i.e filling the experiance replay bugffer)
            # if agent is active, check if the task time is over, if so make the task done and agent available
            # if agent is not available, pass
            if agent["status"] == env.AGENT_ACTIVE:
                # if agent is doing a task, then send the current task (action) but with updated time step to the step function
                next_state, reward, done, truncated, info = env.step(agent_id, current_action, time_step)
            else:
                # if agent is available, do an observation, decide on an action (select a task) and do it on env
                print(f"===> [collecting training samples - Agent-{env.agents.loc[index,'agent_id']}] is free and hence getting the observation at time step: {time_step}")
                obs = env.get_observation()

                # Choose an action based on the observation (POLICY)
                action =  choose_random_action(obs['tasks'])  # this is where we should put out model (i.e give the obs, and get an action to do)
                if action is None:
                    print(f"===> [collecting training samples - Agent-{env.agents.loc[index,'agent_id']}] could not find any available tasks to do at time step: {time_step}, hence skipping to next agent\n")
                    continue

                print(f"===> [collecting training samples - Agent-{env.agents.loc[index,'agent_id']}] selected the task: {action} at time step: {time_step}")

                # do the action on environment and get the next state, reward, done, truncated and info
                next_state, reward, done, truncated, info = env.step(agent_id, action, time_step)

                # store the experiance in the replay buffer if only the agent is the training agent
                if agent_id == training_agent_id:
                    # buffer.store(obs, action -1, reward, next_state, done)

                    ## store only task_type, order, status
                    task_obsveration_for_dqn = obs['tasks']

                    # if task status is DONE, update all values in the task with 0
                    for i, task in enumerate(task_obsveration_for_dqn):
                        if task[env.TASK_STATUS] == env.DONE:
                            task_obsveration_for_dqn[i] = [0, 0, 0, 0, 0, 0, 0, 0]

                    task_obsveration_for_dqn = np.array(task_obsveration_for_dqn)
                    task_obsveration_for_dqn = task_obsveration_for_dqn[:, [env.TASK_TYPE, env.TASK_ORDER, env.TASK_STATUS]]
                    
                    # normalize the task observation
                    task_obsveration_for_dqn = np.array(task_obsveration_for_dqn, dtype=float)
                    # print(data.shape[1])

                    # for col_index in range(task_obsveration_for_dqn.shape[1]):
                    #     # print(col_index)
                    #     col = task_obsveration_for_dqn[:, col_index]
                    #     min_val = np.min(col)
                    #     max_val = np.max(col)
                    #     task_obsveration_for_dqn[:, col_index] = (col - min_val) / (max_val - min_val)

                    task_obsveration_for_dqn =  task_obsveration_for_dqn.tolist() # convert back to a list from a numpy array

                    ### store only task_type, order, status for next_state as well
                    next_state_task_obsveration_for_dqn = next_state['tasks']

                    # if task status is not AVAILABLE, update all values in the task with 0
                    for i, task in enumerate(next_state_task_obsveration_for_dqn):
                        if task[env.TASK_STATUS] == env.DONE:
                            next_state_task_obsveration_for_dqn[i] = [0, 0, 0, 0, 0, 0, 0, 0]
                    
                    next_state_task_obsveration_for_dqn = np.array(next_state_task_obsveration_for_dqn)
                    next_state_task_obsveration_for_dqn = next_state_task_obsveration_for_dqn[:, [env.TASK_TYPE, env.TASK_ORDER, env.TASK_STATUS]]
                    
                    # normalize the task observation
                    next_state_task_obsveration_for_dqn = np.array(next_state_task_obsveration_for_dqn, dtype=float)
                    # print(data.shape[1])

                    # for col_index in range(next_state_task_obsveration_for_dqn.shape[1]):
                    #     # print(col_index)
                    #     col = next_state_task_obsveration_for_dqn[:, col_index]
                    #     min_val = np.min(col)
                    #     max_val = np.max(col)
                    #     next_state_task_obsveration_for_dqn[:, col_index] = (col - min_val) / (max_val - min_val)

                    next_state_task_obsveration_for_dqn =  next_state_task_obsveration_for_dqn.tolist() # convert back to a list from a numpy array
                
                
                    buffer.store(task_obsveration_for_dqn, action -1, reward, next_state_task_obsveration_for_dqn, done)
                    # buffer.store(obs, action -1, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    ## End of experiance replay ###
                
                # only train if the agent is the training agent and the buffer is enough (collected enough samples)
                if agent_id == training_agent_id and len(buffer.buffer) > batch_size:
                    print("\n ===> [Training]: finished collecting training sample (filled experiance replay buffer)")
                    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                    # print('\n====>[Training]: states:', states, 'actions:', actions, 'rewards:', rewards, 'next_states:', next_states, 'dones:', dones, '\n')
                    print("\n ===> [Training]: len(states):", len(states), ",len(actions):", len(actions), ",len(rewards):", len(rewards), ",len(next_states):", len(next_states), ",len(dones):", len(dones))
                    print("actions:", actions)
                    print("actions[0]:", actions[0])
                    
                    # # get a batch of tasks and devices
                    # tasks = [[sublist for sublist in element['tasks']] for element in states]
                    # devices = [[sublist for sublist in element['devices']] for element in states]

                    # # Extend devices with dummy not usable devices. This is needed so that we can input to the model.
                    # # note that here devices means a batch of devices lists (i.e this is a 3D list)
                    # for device_list in devices: # Loop through the 3D list
                    #     for i in range(len(device_list) + 1, env.action_space.n + 1):
                    #         new_inner_list = [i, env.NOT_A_DEVICE, env.ACTIVE, 999]
                    #         device_list.append(new_inner_list)


                    # get a batch of next tasks and devices
                    # next_tasks = [[sublist for sublist in element['tasks']] for element in next_states]
                    # next_devices = [[sublist for sublist in element['devices']] for element in next_states]

                    # # Extend next_devices with dummy not usable devices. This is needed so that we can input to the model.
                    # # note that here devices means a batch of devices lists (i.e this is a 3D list)
                    # for next_device_list in next_devices: # Loop through the 3D list
                    #     for i in range(len(next_device_list) + 1, env.action_space.n + 1):
                    #         new_inner_list = [i, env.NOT_A_DEVICE, env.ACTIVE, 999]
                    #         next_device_list.append(new_inner_list)

                    print("\n===> [Training]: the first observation in the selected batch from experiances for training =====>")
                    print("[Training]: tasks[0]:", states[0])
                    # print("\n[Training]: devices[0]:", devices[0])

                    # Convert to TensorFlow tensors
                    tasks_tensor = tf.convert_to_tensor(states, dtype=tf.float32)  # Specify float32 for common use 
                    # devices_tensor = tf.convert_to_tensor(devices, dtype=tf.float32)

                    next_tasks_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
                    # next_devices_tensor = tf.convert_to_tensor(next_devices, dtype=tf.float32)

                    # Calculate Q-values from the main network
                    q_values_from_main_network = dqn([tasks_tensor])  # Initialize targets with current Q-values (shape is: batch_size, 1, num_actions)
                    q_values_from_main_network = tf.reshape(q_values_from_main_network, (batch_size, env.action_space.n)) # Reshaping the EagerTensor
                    print("\n===> [Training]: Q values from main network by passing a batch of experiances: =====>\n")
                    print(q_values_from_main_network[0])
                    # print(q_values_from_main_network[1][7])

                    # Get Q-values from target network
                    q_values_from_target_network = target_dqn([next_tasks_tensor])
                    q_values_from_target_network = tf.reshape(q_values_from_target_network, (batch_size, env.action_space.n)) # Reshaping the EagerTensor
                    print("\n===> [Training]: Q values from target network by passing a batch of experiances: =====>\n")
                    print("q_values_from_target_network[1]: ",q_values_from_target_network[1])
                    # print("q_values_from_target_network[2]: ",q_values_from_target_network[2])


                    # Calculate the maximum Q-values of the next states (from the target network)
                    max_q_values_of_next_states = np.max(q_values_from_target_network, axis=1)
                    print("\n===> [Training]: Max Q values from target network by passing a batch of experiances: =====>\n")
                    print(max_q_values_of_next_states)

                    # Calculate the expected future rewards
                    real_expected_future_rewards = q_values_from_main_network.numpy()  # Initialize with current Q-values from main network and convert to numpy array
                    real_expected_future_rewards[range(batch_size), actions - 1] = rewards + (1 - dones) * gamma * max_q_values_of_next_states
                    print("\n===> [Training]: Q values from main network, after updating the taken action with real expected future reward: =====>\n")
                    print(real_expected_future_rewards)

                    # Train the DQN
                    with tf.GradientTape() as tape:
                        q_values = dqn([tasks_tensor])
                        loss = tf.keras.losses.mean_squared_error(real_expected_future_rewards, q_values)
                        grads = tape.gradient(loss, dqn.trainable_variables)
                        optimizer.apply_gradients(zip(grads, dqn.trainable_variables))

                        losses.append(np.mean(loss[0].numpy()))

    # Periodically update the target network
    if episode % 10 == 0:
        print("\n===> [Training]: Updating the target network weights with the main network weights ========= \n")
        target_dqn.set_weights(dqn.get_weights()) 


# save the model
print("\n===> [Training]: Saving the model ========= \n")
dqn.save('./dqn_model.keras')

print("\n===> [Training]: Model saved successfully ========= \n")
dqn.save('./dqn_model.h5')

# load the model
print("\n===> [Training]: Loading the model ========= \n")
model = keras.models.load_model('dqn_model.h5')

# plot the losses
print("\n===> [Training]: Plotting the losses ========= \n")
print(losses)

import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.show()

