import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import random

from WarehouseEnv_V0 import WarehouseEnv
env = WarehouseEnv()

# Simple agent for demonstration
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.task_id = None
        self.task_start_time_step = 0
        self.device_id = None

    def act(self, observation):
        return self.action_space.sample()

# Helper function to choose an action based on the current state
def choose_random_action(tasks):
    # Choose an random task (action). But this task should be an available task
    available_tasks = [x for x in tasks if x[env.TASK_STATUS] == 0]  # Filter tasks with status 0 (available)
    available_task_ids = [sublist[0] for sublist in available_tasks]
    random_action = random.choice(available_task_ids) # Randomly select an available task. (task and device ids start from 1)
    print(f"Random action selected:", random_action)
    return random_action


######################################################
############# --- Main Interaction ---  #############
#######################################################

# Note: tasks and devices will be read, encoded and handled in the WarehouseEnv class

# agents
agents = pd.read_csv('agents.csv')
agent_type_encodings = {
    "human": 0,
    "robot": 1
}
agents["type"] = agents["type"].map(agent_type_encodings)

agent_status_encoding = {
    "available": 0,
    "active": 1,
}
agents["status"] = agents["status"].map(agent_status_encoding)
print("===> [main]: agents.head(): \n", agents.head(10))

# Convert all columns that can be successfully converted to int
for col in agents.columns:
    # agents[col] = agents[col].astype('int')
    agents[col] = agents[col].apply(pd.to_numeric, errors='coerce')


# A dataframe to store the task started time, who did it and when it is done
# this will be used to create a report and visualize the performance of the agents
task_report = pd.DataFrame(columns=["task_id", "agent_id", "device_id", "task_status", "task_time", "start_time", "end_time"])
agent_activity_report = pd.DataFrame(columns=["agent_id", "task_id", "device_id", "start_time", "end_time"])

#############

done = False
time_step = -1

while not done:
    time_step = time_step + 1

    # if time_step > 11:
    #     done = True
    #     break

    for index, agent in agents.iterrows():
        print(f"===> [main - Agent-{agent['agent_id']}] turn at time step: {time_step}")
        print(f"===> [main - Agent-{agent['agent_id']}] current_task: {agent['current_task']} \
              current_device: {agent['current_device']}, status: {agent['status']}, type: {agent['type']}")

        task_index = agent["current_task"] -1 if agent["current_task"] != -1 else -1
        agent_device = agent['current_device']
        agent_device_index = agent_device - 1
        agent_type = agent['type']


        # if agent is not available, pass
        if agent["status"] == env.AGENT_ACTIVE:
            # # if more time there to complete task, pass
            if time_step <= env.tasks[task_index][env.TASK_TIME] + task_report[task_report["task_id"] == agent['current_task']]["start_time"].values[0]:
                print(f"===> [main - Agent-{agent['agent_id']}] is still doing the task: {agent['current_task']} at time step: {time_step}, hence skipping to next agent\n")
                continue # pass and continue to next agent
            else:
                print(f"===> [main - Agent-{agents.loc[index,'agent_id']}] is done with the task: {agents.loc[index, 'current_task']} at time step: {time_step}")

                # make the agent free
                agents.loc[index, "status"] = env.AGENT_AVAILABLE
                agents.loc[index, 'current_task'] = -1
                agents.loc[index, 'current_device'] = -1

                # make the task done
                env.tasks[task_index][env.TASK_STATUS] = env.DONE

                # free the device
                env.devices[agent_device_index][env.DEVICE_STATUS] = env.AVAILABLE
                env.devices[agent_device_index][env.DEVICE_CURRENT_TASK_ID] = -1
                

                # update the task report
                task_report.loc[task_report["task_id"] == agent['current_task'], "end_time"] = time_step
                task_report.loc[task_report["task_id"] == agent['current_task'], "task_status"] = env.DONE

                # update the agent activity report
                agent_activity_report.loc[agent_activity_report["agent_id"] == agent['agent_id'], "end_time"] = time_step

                print(f"===> [main - Agent-{agents.loc[index, 'agent_id']}] is in {agents.loc[index, 'status']} state at time step: {time_step}")

        # if agent is available, do an observation and take an action (select a task)
        if agents.loc[index, "status"] == env.AGENT_AVAILABLE:
            print(f"===> [main - Agent-{agents.loc[index,'agent_id']}] is available at time step: {time_step}")
            agents.loc[index,'status'] = env.AGENT_ACTIVE

            # get the observation for the agent
            obs = env.get_observation()

            # Choose and action based on the observation (POLICY)
            action =  choose_random_action(obs['tasks'])  # this is where we should put out model (i.e give the obs, and get an action to do)
            task_index = action - 1

            # do the action on environment and get the next state, reward, done, truncated and info
            next_state, reward, done, truncated, info = env.step(task_index, agent_type)

            # Since enviorenment will decide the device (based on task and device availability)
            # we have get the device ID once the action is performed on the environment
            for i, row in enumerate(next_state['devices']):
                if row[env.DEVICE_CURRENT_TASK_ID] == action:
                    device_id = row[env.DEVICE_ID]
                    break     
            device_index = device_id - 1 # Because devices are a 2D array and we select them based on the index of inner array

            # update agent attributes to indicate start of assigned task and to keep track of task_id, device, reward and etc
            agents.loc[index,'current_task'] = action
            agents.loc[index,'current_device'] = device_id
            agents.loc[index,'reward'] = agents.loc[index, 'reward'] + reward

            print(f"===> [main - Agent-{agents.loc[index,'agent_id']}] started doing the task: {agents.loc[index, 'current_task']} at time step: {time_step}, on device: {device_id}\n")

            # add entries to the task report and agent activity report
            task_new_record = pd.DataFrame([[action, agents.loc[index,'agent_id'], device_id, env.ACTIVE, env.tasks[task_index][env.TASK_TIME], time_step, None]], columns=task_report.columns)
            task_report = pd.concat([task_report, task_new_record], ignore_index=True)

            agent_activity_new_record = pd.DataFrame([[agents.loc[index,'agent_id'], action, device_id, time_step, None]], columns=agent_activity_report.columns)
            agent_activity_report = pd.concat([agent_activity_report, agent_activity_new_record], ignore_index=True)


print("===> [main]: task_report.head(): \n", task_report.head(10))
print("===> [main]: agent_activity_report.head(): \n", agent_activity_report.head(10))
print("===> [main]: agents.head(): \n", agents.head(10))

