import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

from WarehouseEnv_V0 import WarehouseEnv

# Simple agent for demonstration
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.task_id = None
        self.task_start_time_step = 0
        self.device_id = None

    def act(self, observation):
        return self.action_space.sample()


######################################################
############# --- Main Interaction ---  #############
#######################################################
tasks = pd.read_csv('tasks.csv')
devices = pd.read_csv('devices.csv')
tasks.head()

# encode "type"
type_encoding = {
    "pick": 0,
    "put": 1,
    "load": 2,
    "repl": 3
}
tasks["type"] = tasks["type"].map(type_encoding)

tasks["product"] = tasks["product"].apply(lambda prod:  prod[1:]) # encode product
tasks["from loc"] = tasks["from loc"].apply(lambda loc:  loc[1:]) # encode from loc
tasks["to loc"] = tasks["to loc"].apply(lambda loc:  loc[1:]) # encode to loc

tasks.fillna({'order':'o0'}, inplace=True)  # encode to order
tasks["order"] = tasks["order"].apply(lambda order:  order[1:])


# encode status
status_encoding = {
    "available": 0,
    "active": 1,
    "done": 2,
}
tasks["status"] = tasks["status"].map(status_encoding)

tasks.head()

task_list = tasks.to_numpy().tolist()
task_list = [list( map(int,i) ) for i in task_list]
print(task_list)

devices = pd.read_csv('devices.csv')
# devices.head()

# type encodings
type_encodings = {
    "forklift": 1,
    "pallet_jack": 0
}
devices["type"] = devices["type"].map(type_encodings)

# encode status
status_encoding = {
    "available": 0,
    "active": 1,
}
devices["status"] = devices["status"].map(status_encoding)

device_list = devices.to_numpy().tolist()
print(device_list)

devices.head(10)

#########

env = WarehouseEnv(tasks=task_list, devices=device_list)

agents = [RandomAgent(env.action_space), RandomAgent(env.action_space)]

num_episodes = 5
for episode in range(num_episodes):
    obs, info = env.reset()

    done = False
    time_step = 0 # within episode time step (not the episode number)

    while not done:
        time_step = time_step + 1
        for agent in agents:

            # if the agent is doing a task and current_time_step is grater than task time + task_start_time_step
            # make that task done and agent is free to take a new task.
            if agent.task_id is not None:
                # if more time there to complete task, pass
                if time_step <= env.tasks[agent.task_id][env.TASK_TIME] + agent.task_start_time_step:
                    continue
                else:
                    # make agent free and update the task to done and device status to available (if any)
                    env.tasks[agent.task_id][env.TASK_STATUS] = env.DONE
                    env.devices
                    agent.task_id = None

            else:
                action = agent.act(obs)  # this is where we should put out model (i.e give the obs, and get an action to do)

                # update agent attributes to indicate start of assigned task and to keep track of task_id, start time and etc
                agent.task_id = action
                agent.task_start_time_step = time_step

                obs, reward, done, trunc, info = env.step(action)

            print(f"Agent action: {action}, Reward: {reward}, Task List: {obs} \n") 