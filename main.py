import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import random
from keras.models import load_model
import tensorflow as tf


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
    print("[choosing random task]: input to choose_random_action: \n", tasks)
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


######################################################
############# --- Main Interaction ---  #############
#######################################################

# Note: tasks, devices and agents will be read, encoded and handled in the WarehouseEnv class

done = False
time_step = -1

time_step_records = pd.DataFrame(columns=['time_step', 'remaining_tasks', 'free_agents', 'free_devices'])
task_completion_records = pd.DataFrame(columns=['time_step', 'pick', 'repl', 'put', 'load'])
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

    # Tract remaining task types at each time step and track them in task_completion_records
    remaining_pick_tasks = sum(task[env.TASK_TYPE] == env.PICK for task in env.tasks if task[env.TASK_STATUS] == env.AVAILABLE)
    remaining_repl_tasks = sum(task[env.TASK_TYPE] == env.REPL for task in env.tasks if task[env.TASK_STATUS] == env.AVAILABLE)
    remaining_put_tasks = sum(task[env.TASK_TYPE] == env.PUT for task in env.tasks if task[env.TASK_STATUS] == env.AVAILABLE)
    remaining_load_tasks = sum(task[env.TASK_TYPE] == env.LOAD for task in env.tasks if task[env.TASK_STATUS] == env.AVAILABLE)
    task_completion_rec = pd.DataFrame([[time_step, remaining_pick_tasks, remaining_repl_tasks, remaining_put_tasks, remaining_load_tasks]], columns=task_completion_records.columns)
    task_completion_records = pd.concat([task_completion_records, task_completion_rec], ignore_index=True)

    print(f"\n=====> [main] : start of time step: {time_step}")
    for index, agent in env.agents.iterrows(): 
        agent_id = agent['agent_id']

        current_action = agent["current_task"]
        agent_device = agent['current_device']
        agent_device_index = agent_device - 1
        agent_type = agent['type']


        # if agent is not available, pass
        if agent["status"] == env.AGENT_ACTIVE:
            # if agent is doing a task, then send the current task (action) but with updated time step to the step function
            print(f"\n===> [main - Agent-{env.agents.loc[index,'agent_id']}] is busy with task: {current_action} at time step: {time_step}")
            next_state, reward, done, truncated, info = env.step(agent_id, current_action, time_step)

        else:
            # if agent is available, do an observation, decide on an action (select a task) and do it on env
            # get the observation for the agent
            print(f"\n===> [main - Agent-{env.agents.loc[index,'agent_id']}] is free and hence getting the observation at time step: {time_step}")
            obs = env.get_observation()

            observed_tasks = obs['tasks']
            observed_devices = obs['devices']

            # Choose and action based on the observation (POLICY)
            # if agent uses dqn model (as policy) to choose an action use it here
            if agent_id in env.DQN_AGENTS:
                # load the keras dqn model called dqn_model.keras and use it to choose an action
                model = load_model('dqn_model.keras')

                # prepare the observation for the model
                # Extend devices with dummy not usable devices. This is needed so that we can input to the model.
                # note that here devices means a lists of devices (i.e this is a 2D list)
                # for i in range(len(observed_devices) + 1, env.action_space.n + 1):
                #     new_dummy_device = [i, env.NOT_A_DEVICE, env.ACTIVE, 999]
                #     observed_devices.append(new_dummy_device)

                # convert the observed tasks and devices to numpy arrays
                observed_tasks_for_dqn = np.array(obs['tasks'])
                # observed_devices = np.array(observed_devices)

                # if task status is Done, update all values in the task with 0
                for i, task in enumerate(observed_tasks_for_dqn):
                    if task[env.TASK_STATUS] == env.DONE:
                        observed_tasks_for_dqn[i] = [0, 0, 0, 0, 0, 0, 0, 0]

                ## store only task_id, task_type, order, status
                observed_tasks_for_dqn = observed_tasks_for_dqn[:, [env.TASK_TYPE, env.TASK_ORDER, env.TASK_STATUS]]
                
                # normalize the task observation
                observed_tasks_for_dqn = np.array(observed_tasks_for_dqn, dtype=float)
                # print(data.shape[1])

                # for col_index in range(observed_tasks_for_dqn.shape[1]):
                #     # print(col_index)
                #     col = observed_tasks_for_dqn[:, col_index]
                #     min_val = np.min(col)
                #     max_val = np.max(col)
                #     observed_tasks[:, col_index] = (col - min_val) / (max_val - min_val)

                observed_tasks_for_dqn =  observed_tasks_for_dqn.tolist() # convert back to a list from a numpy array


                observed_tasks_for_dqn = np.expand_dims(observed_tasks_for_dqn, axis=0)  # Add the batch axis
                # observed_devices = np.expand_dims(observed_devices, axis=0) # Add the batch axis

                # convert to tensors
                observed_tasks_for_dqn = tf.convert_to_tensor(observed_tasks_for_dqn)
                # observed_devices = tf.convert_to_tensor(observed_devices)

                # pass and get the prediction
                print("======> [main] input to model: \n",observed_tasks_for_dqn)
                q_values_from_main_network = model([observed_tasks_for_dqn]) 
                print("======> [main] output from model: ",q_values_from_main_network)

                # choose the action with the highest q value
                action = np.argmax(q_values_from_main_network)
                print(f"===> [main - Agent-{env.agents.loc[index,'agent_id']}] selected the task index: {action} at time step: {time_step}")

                action =  action + 1 # increment by 1 to match the task id (task ids start from 1)
   
            else:
                action =  choose_random_action(observed_tasks)  # this is where we should put out model (i.e give the obs, and get an action to do)
            
            if action is None:
                print(f"===> [main - Agent-{env.agents.loc[index,'agent_id']}] could not find any available tasks to do at time step: {time_step}, hence skipping to next agent\n")
                continue

            print(f"===> [main - Agent-{env.agents.loc[index,'agent_id']}] selected the task: {action} at time step: {time_step}")

            # do the action on environment and get the next state, reward, done, truncated and info
            next_state, reward, done, truncated, info = env.step(agent_id, action, time_step)
        
    if time_step == 400:
        break


print("Simulation done!")
print("Task report:")
# if there are records where end time is None, then set the end time to the start time + task time
# loop through the task report and update the end time for the tasks that are not done
for index, row in env.task_report.iterrows():
    if row["end_time"] is None:
        env.task_report.loc[index, "end_time"] = row["start_time"] + row["task_time"]

print(env.task_report)

# saving the env tasks
final_task_status = pd.DataFrame(columns=['task_id','type','product','qty','from loc','to loc','time','order','status'])
print("Env Tasks after simulation done:")
for row in env.tasks:  # Iterate over rows
    task_rec = pd.DataFrame([[
        row[env.TASK_ID], row[env.TASK_TYPE], row[env.TASK_PRODUCT], row[env.TASK_QTY], row[env.TASK_FROM_LOC], row[env.TASK_TO_LOC],\
        row[env.TASK_TIME], row[env.TASK_ORDER], row[env.TASK_STATUS]
    ]]
    )
    final_task_status = pd.concat([final_task_status, task_rec], ignore_index=True)

# save final_task_status
final_task_status.to_csv("final_task_status.csv", index=False)


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


# plot the task completion records
plt.figure()
plt.plot(task_completion_records["time_step"], task_completion_records["pick"], label="Pick Tasks")
plt.plot(task_completion_records["time_step"], task_completion_records["repl"], label="Repl Tasks")
plt.plot(task_completion_records["time_step"], task_completion_records["put"], label="Put Tasks")
plt.plot(task_completion_records["time_step"], task_completion_records["load"], label="Load Tasks")
plt.xlabel("Time Step")
plt.ylabel("Count")
plt.title("Task Completion Timeline")
plt.legend()

plt.show()

# save task_completion_records to a csv file
task_completion_records.to_csv("task_completion_records.csv", index=False)


