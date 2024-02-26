import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class WarehouseEnv(gym.Env):
    """
    This a custom enviorenment that follows gym interface.
    This is the V0 of a simple warehouse env for an agent. 
    
    Designing a gym env means, designing a env so that an agent can use
    rest(), step() kind of methods. An env is not a agent. env is the 
    place where agents work (act on). 
    
    So no internal agent level attributes should be retained in this env class.

    Agent waits the relevent time steps of the tasks before taking on another task.
    by this, we can do some visualizations too.   

    """

    metadata = {'render.modes': ['console']}

    # Define constants for clear code
    REST = 0
    ACTIVE = 1
    HUMAN = 0
    ROBOT = 1

    #task_list_columns
    TASK_ID = 0
    TASK_TYPE = 1
    TASK_PRODUCT = 2
    TASK_FROM_LOC = 3
    TASK_TO_LOC = 4
    TASK_TIME = 5
    TASK_ORDER = 6
    TASK_STATUS = 7

    # Task types
    PICK = 0
    PUT = 1
    LOAD = 2
    REPL = 3

    # task / device status
    AVAILABLE = 0
    ACTIVE = 1
    DONE = 2

    # DEVICE COLUMNS
    DEVICE_ID = 0
    DEVICE_TYPE = 1
    DEVICE_STATUS = 2

    # DEVICE TYPE
    PALLET_JACK = 0
    FORKLIFT = 1
   
    def __init__(self, tasks=None, devices=None):
        super(WarehouseEnv, self).__init__()

        # The action of a agent is selecting a task_id to do.
        # So, action state can be a large number. That means the number of available tasks 
        # can be anything which will be provided in tasks list. 

        # following are the observations that an agent will see from the env
        self.tasks = np.array(tasks) # an observation
        self.devices = np.array(devices) # an observation

        self.action_space = spaces.Discrete(len(tasks)); # assuming at a given time max number of actions is 100.

        # OBSERVATIONS FROM ENV ARE: warehouse tasks, devices
        
        # Warehouse Tasks
        self.task_shape = (len(self.tasks), 8)  # Adjust if you have more task attributes
        self.task_low = np.array([[
            0,  # Task_Id (assume non-negative integers)
            0,  # Type (might be categorical - see below)
            0,  # Product
            0,  # From Location
            0,  # To Location
            0,  # Time (might be a float)
            0,  # Order No
            0   # Status (might be categorical - see below)
        ] for _ in range(len(tasks))])
        self.task_high = np.array([[
            100,  # Task_Id
            4,  # Type (if categorical, maximum category value)
            100,  # Product
            100,  # From Location 
            100,  # To Location
            60,   # Time (if assuming a 24-hour window) 
            100,  # Order No
            3   # Status 
        ] for _ in range(len(tasks))])

        # Warehouse Devices
        self.device_shape = (len(self.devices),3)  # Adjust if you have more device attributes
        self.device_low = np.array([[
            0,  # Device_Id
            0,  # Type 
            0   # Status
        ] for _ in range(len(devices))])
        self.device_high = np.array([[
            100,  # Device_Id
            2,  # Type
            2  # Status
        ] for _ in range(len(devices))])

        # print(self.task_low)

        # Combining spaces using dictionaries
        self.observation_space = spaces.Dict({
            'tasks': spaces.Box(low=self.task_low, high=self.task_high, shape=np.array(self.tasks).shape, dtype=np.int32),
            'devices': spaces.Box(low=self.device_low, high=self.device_high, shape=np.array(self.devices).shape, dtype=np.int32),
        })
    
    # This generates random sample observations (for testing).
    # You'll replace this with logic to fetch actual warehouse data
    def sample(self):
        # Example sampling (you'll replace this with  your actual data)
        return {
            'tasks': np.random.randint(self.task_low, self.task_high, size=np.array(self.tasks).shape),
            'devices': np.random.randint(self.device_low, self.device_high, size=np.array(self.devices).shape)
        } 


    def reset(self, seed=0.0):
        """
        Importent: the observation must be a a tuple containing 
        observation and a info object.

        i.e return: (obs, info)
        """

        # initialize observations

        # Althouhg __init__() method received the tasks from the main method, here in
        # reset() method we cannot relly on that. because after each iterations, the
        # self.tasks will get updated. So it no longer will have the initial task list 
        # which was supplied at begining. 
        tasks = pd.read_csv('tasks.csv')

        # encode the df
        type_encoding = {"pick": self.PICK, "put": self.PUT, "load": self.LOAD, "repl": self.REPL} # encode "type"
        tasks["type"] = tasks["type"].map(type_encoding)
        tasks["product"] = tasks["product"].apply(lambda prod:  prod[1:]) # encode product
        tasks["from loc"] = tasks["from loc"].apply(lambda loc:  loc[1:]) # encode from loc
        tasks["to loc"] = tasks["to loc"].apply(lambda loc:  loc[1:]) # encode to loc
        tasks.fillna({'order':'o1'}, inplace=True)  # encode to order
        tasks["order"] = tasks["order"].apply(lambda order:  order[1:])     
        status_encoding = {"available": self.AVAILABLE, "active": self.ACTIVE, "done": self.DONE} # encode status
        tasks["status"] = tasks["status"].map(status_encoding)

        task_list = tasks.to_numpy().tolist()
        task_list = [list( map(int,i) ) for i in task_list]

        
        self.tasks = task_list # Initialize or reset the task_list to its starting state. 
                                # Likely, all tasks would have the status "Available".
        
        
        # Similarly Get devices from CSV itself
        devices = pd.read_csv('devices.csv')
      
        type_encodings = {"forklift": self.FORKLIFT,"pallet_jack": self.PALLET_JACK} # type encodings
        devices["type"] = devices["type"].map(type_encodings)
        status_encoding = {"available": self.AVAILABLE, "active": self.ACTIVE} # encode status
        devices["status"] = devices["status"].map(status_encoding)

        device_list = devices.to_numpy().tolist()
        self.devices = device_list

        # the whole observation object:
        observation = {
            "tasks": np.array(task_list),
            "devices": np.array(device_list)
        }

        info = {
            "tasks": self.tasks,
            "devices": self.devices  
        }

        return (observation, info)


    def step(self, action, agent_type=HUMAN):
        # Here, an action will be a task_id and based on task status we can return a reward.
        print("action:", action, "agent_type", agent_type)

        if action >= len(self.tasks):
            raise ValueError("Received a task id (as the action) grater than available number of tasks")

        print("action number is a valid one")
        print(self.tasks[1][7])
        print(self.AVAILABLE)


        # REWARD CALCULATION
        # Punish if the selected task is alreay active (Assigned) or completed (done)
        if self.tasks[action][self.TASK_STATUS] != self.AVAILABLE:
            reward = -1
        else:
            # if agent is human
            if agent_type == self.HUMAN:

                print("in human reward calculation 1")

                # if task is a pick, a pallet_jack is needed. 
                if self.tasks[action][self.TASK_TYPE] == self.PICK:
                    print("in human reward calculation 2")
                    pallet_jacks = [device for device in self.devices if device[self.DEVICE_TYPE] == self.PALLET_JACK]
                    num_available_pallet_jacks = len([pj for pj in pallet_jacks if pj[self.DEVICE_STATUS] == self.AVAILABLE])

                    if num_available_pallet_jacks > 0:
                        reward = 1

                        # make the task active
                        self.tasks[action][self.TASK_STATUS] = self.ACTIVE

                        # make that pallet status as active (the first pj which is available)
                        for index, device in enumerate(self.devices):
                            if device[self.DEVICE_TYPE] == self.PALLET_JACK and device[self.DEVICE_STATUS] == self.AVAILABLE:
                                self.devices[index][self.DEVICE_STATUS] = self.ACTIVE
                                break
                    else:
                        reward = -1 # punish if no available pallet jacks
                else: #every other task types needs a forklift 
                    print("in human reward calculation 3")
                    forklifts = [device for device in self.devices if device[self.DEVICE_TYPE] == self.FORKLIFT]
                    num_available_forklifts = len([fk for fk in forklifts if fk[self.DEVICE_STATUS] == self.AVAILABLE])

                    print(num_available_forklifts)

                    if num_available_forklifts > 0:
                        reward = 1

                        # make the task active
                        self.tasks[action][self.TASK_STATUS] = self.ACTIVE

                        # make that forklift status as active (the first fk which is available)
                        for index, device in enumerate(self.devices):
                            if device[self.DEVICE_TYPE] == self.FORKLIFT and device[self.DEVICE_STATUS] == self.AVAILABLE:
                                self.devices[index][self.DEVICE_STATUS] = self.ACTIVE
                                break
                    else:
                        reward = -1 # punish if no available pallet jacks
                    print("in human reward calculation 3.3")

            else: # robots can only do pick tasks and they dont need any devices for that. 

                if self.tasks[action][self.TASK_TYPE] != self.PICK: # if the task type is not a pick task, punish
                    reward = -1
                else: 
                    reward = 1
                    
                    # make the task to active status
                    self.tasks[action][self.TASK_STATUS] = self.ACTIVE

        print("reward: ", reward)
        # DONE?
        num_available_tasks = len([task for task in self.tasks if task[self.TASK_STATUS] == self.AVAILABLE])
        if num_available_tasks > 0:
            done = False
        else:
            done = True

        print("done: ", done)

        # TRUNCATED?
        truncated = False

        print("truncated: ", truncated)
        

        # INFO?
        info = {}

        return self.tasks, reward, done, truncated, info
    

    def render(self, mode='console'):
        print("within render method")
    
    def close(self):
        pass





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

env = WarehouseEnv(tasks=task_list, devices=device_list)
print("env taskssss===")
print(env.tasks)
print(env.devices)

print(env.observation_space)

print("a sampleeeee: ")
print(env.sample())

print("reseting ======")
print(env.reset())


print("action space")
print(env.action_space)
print(env.action_space.sample())

print("take a action to select task 1 by a human====")
print(env.step(1, env.HUMAN))
        
