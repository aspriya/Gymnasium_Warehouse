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

    if len(available_task_ids) > 0:
        random_action = random.choice(available_task_ids) # Randomly select an available task. (task and device ids start from 1)
        print(f"Random action selected:", random_action)
        return random_action
    else:
        print("No available tasks to choose from. Returning None.")
        return None


######################################################
############# --- Main Interaction ---  #############
#######################################################

# Note: tasks, devices and agents will be read, encoded and handled in the WarehouseEnv class


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

    for index, agent in env.agents.iterrows():
        agent_id = agent['agent_id']

        current_action = agent["current_task"]
        agent_device = agent['current_device']
        agent_device_index = agent_device - 1
        agent_type = agent['type']


        # if agent is not available, pass
        if agent["status"] == env.AGENT_ACTIVE:
            # if agent is doing a task, then send the current task (action) but with updated time step to the step function
            next_state, reward, done, truncated, info = env.step(agent_id, current_action, time_step)


            # # update the task report
            # task_report.loc[task_report["task_id"] == agent['current_task'], "end_time"] = time_step
            # task_report.loc[task_report["task_id"] == agent['current_task'], "task_status"] = env.DONE

            # # update the agent activity report
            # agent_activity_report.loc[agent_activity_report["agent_id"] == agent['agent_id'], "end_time"] = time_step

        # if agent is available, do an observation and take an action (select a task)
        else:
            # get the observation for the agent
            print(f"===> [main - Agent-{env.agents.loc[index,'agent_id']}] is free and hence getting the observation at time step: {time_step}")
            obs = env.get_observation()

            # Choose and action based on the observation (POLICY)
            action =  choose_random_action(obs['tasks'])  # this is where we should put out model (i.e give the obs, and get an action to do)
            if action is None:
                print(f"===> [main - Agent-{env.agents.loc[index,'agent_id']}] could not find any available tasks to do at time step: {time_step}, hence skipping to next agent\n")
                continue

            print(f"===> [main - Agent-{env.agents.loc[index,'agent_id']}] selected the task: {action} at time step: {time_step}")

            # do the action on environment and get the next state, reward, done, truncated and info
            next_state, reward, done, truncated, info = env.step(agent_id, action, time_step)


            # add entries to the task report and agent activity report
            # task_new_record = pd.DataFrame([[action, env.agents.loc[index,'agent_id'], device_id, env.ACTIVE, env.tasks[task_index][env.TASK_TIME], time_step, None]], columns=task_report.columns)
            # task_report = pd.concat([task_report, task_new_record], ignore_index=True)

            # agent_activity_new_record = pd.DataFrame([[env.agents.loc[index,'agent_id'], action, device_id, time_step, None]], columns=agent_activity_report.columns)
            # agent_activity_report = pd.concat([agent_activity_report, agent_activity_new_record], ignore_index=True)


print("===> [main]: task_report.head(): \n", task_report.head(10))
print("===> [main]: agent_activity_report.head(): \n", agent_activity_report.head(10))
print("===> [main]: agents.head(): \n", env.agents.head(10))

