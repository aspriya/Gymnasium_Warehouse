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

done = False
time_step = -1

time_step_records = pd.DataFrame(columns=['time_step', 'remaining_tasks', 'free_agents', 'free_devices'])
while not done:
    time_step = time_step + 1

    #update env time step
    env.time_step = time_step

    # if time_step > 11:
    #     done = True
    #     break

    # Create and maintain a df to track no of available tasks, free agents, free devices at each time step.
    # This will be used for analysis and debugging
    available_agents = env.agents.query(f"status == {env.AGENT_AVAILABLE}").shape[0]
    available_devices = len([device for device in env.devices if device[env.DEVICE_STATUS] == env.AVAILABLE])
    time_step_rec = pd.DataFrame([[time_step, \
                        sum(task[env.TASK_STATUS] == env.AVAILABLE for task in env.tasks), \
                        available_agents, \
                        available_devices]], \
                        columns=time_step_records.columns
                    )
    time_step_records = pd.concat([time_step_records, time_step_rec], ignore_index=True)

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

print("Simulation done!")
print("Task report:")
# if there are records where end time is None, then set the end time to the start time + task time
# loop through the task report and update the end time for the tasks that are not done
for index, row in env.task_report.iterrows():
    if row["end_time"] is None:
        env.task_report.loc[index, "end_time"] = row["start_time"] + row["task_time"]

print(env.task_report)

# a df to store agent task distribution
agent_task_distribution = pd.DataFrame(columns=["agent_id", "total_tasks", "total_time", "avg_time_per_task"])
# loop through the agent activity report and calculate the total tasks done by each agent
for agent_id in env.task_report["agent_id"].unique():
    agent_tasks = env.task_report[env.task_report["agent_id"] == agent_id]
    total_tasks = agent_tasks.shape[0]    
    total_time = agent_tasks["task_time"].sum() # sum of all the task times
    avg_time_per_task = total_time / total_tasks

    agent_task_distribution_rec = pd.DataFrame([[agent_id, total_tasks, total_time, avg_time_per_task]], columns=agent_task_distribution.columns)
    agent_task_distribution = pd.concat([agent_task_distribution, agent_task_distribution_rec], ignore_index=True)

agent_task_distribution = agent_task_distribution.sort_values(by="agent_id") # sort by the agent_id
print("Agent task distribution:")
print(agent_task_distribution)

print("Time step records:")
print(time_step_records.tail())

avg_free_agents_per_time_step = time_step_records["free_agents"].mean()
avg_free_devices_per_time_step = time_step_records["free_devices"].mean()
print(f"Average free agents per time step: {avg_free_agents_per_time_step}")
print(f"Average free devices per time step: {avg_free_devices_per_time_step}")

# plot the agent task distribution
import matplotlib.pyplot as plt
plt.bar(agent_task_distribution["agent_id"], agent_task_distribution["total_tasks"])
plt.xlabel("Agent ID")
plt.ylabel("Total Tasks")
plt.title("Agent Task Distribution")

# plot the time step records
plt.figure()
plt.plot(time_step_records["time_step"], time_step_records["remaining_tasks"], label="Remaining Tasks")
plt.plot(time_step_records["time_step"], time_step_records["free_agents"], label="Free Agents")
plt.plot(time_step_records["time_step"], time_step_records["free_devices"], label="Free Devices")
plt.xlabel("Time Step")
plt.ylabel("Count")
plt.title("Time Step Records")
plt.legend()
plt.show()




