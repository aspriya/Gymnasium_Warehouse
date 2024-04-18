import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import random
import pygame
import sys


class WarehouseEnv(gym.Env):
    """
    This a custom enviorenment that follows gym interface.
    This is the V0 of a simple warehouse env for an agent. 
    
    Designing a gymnasium (gym) env means, designing an environment so that an agent can use:
        reset() - reset the env to initial state and return the initial observation, 
        step() - take an action and return the next observation, reward, done, info,
        render() - visualize the env
        sample() - generate a random sample observation  
        close() - close the env and free the resources
    kind of methods. 
    
    An env is not a agent. env is the place where agents work (act on). 
    
    So no internal agent level attributes should be retained in this env class.

    OBSERVATIONS:
    ------------
    An agent observes two things: The tasks and the devices. These should be in csv files and
    should be loaded and then provided to __init()__ after encoding.
    
    Tasks:
    task_id | type | product | qty      | from loc | to loc | time | order | status
    --------|------|---------|----------|----------|--------|------|---------------------
    0       | pick | 2       | 30       | 11       | 21     | 20   |  1     | available
    1       | pick | 3       | 20       | 15       | 22     | 20   |  2     | active
    ........|......|.......  |..........|..........|........|......|........|................
    21      | load | 4       | 50       | 21       | 23     | 20   |        | done
    22      | repl | 5       | 30       |  1       | 11     | 20   |  1     | available
    ....

    Devices:
    device_id | type          | status  | current_task_id
    ----------|---------------|---------|----------------
    0         | pallet_jack   | 0       | NULL
    1         | forklift      | 1       | NULL
    2         | pallet_jack   | 0       | NULL
    ...

    ACTIONS:
    An agent should select a task (no need to learn to select a device, because it is simple given the task type). 
    So the policy should be, an strategy to select a task given the above observation (which is Devices and Tasks).

    Why we should give devices also as an observation? 
        - Because, the agent should know the devices status.
        - An Agent should learn to not to select tasks if no available devices (although there are available tasks).


    ASSUMPTIONS:
    -----------
    1. An Warehouse Agent waits the relevent time steps of the tasks before taking on another task.
    by this, we can do some visualizations too.   

    """

    metadata = {'render.modes': ['console']}

    # Define constants for clear code
    REST = 1
    ACTIVE = -1

    #task_list_columns
    TASK_ID = 0
    TASK_TYPE = 1
    TASK_PRODUCT = 2
    TASK_QTY = 3
    TASK_FROM_LOC = 4
    TASK_TO_LOC = 5
    TASK_TIME = 6
    TASK_ORDER = 7
    TASK_STATUS = 8

    # Task types
    PICK = 1
    PUT = 2
    LOAD = 3
    REPL = 4

    # task / device status
    AVAILABLE = 1
    ACTIVE = -1
    DONE = 0

    # DEVICE COLUMNS
    DEVICE_ID = 0
    DEVICE_TYPE = 1
    DEVICE_STATUS = 2
    DEVICE_CURRENT_TASK_ID = 3

    # DEVICE TYPE
    PALLET_JACK = 1
    FORKLIFT = 2
    NOT_A_DEVICE = 3

    # agent_columns
    AGENT_ID = 0
    AGENT_TYPE = 1
    AGENT_STATUS = 2

    # agent type encodings
    HUMAN = 1
    ROBOT = 2

    # agent status encodings
    AGENT_AVAILABLE = 1
    AGENT_ACTIVE = -1

    # pygame width, height and other attributes
    WIDTH = 1900
    HEIGHT = 1000

    # Right Stat Box properties
    R_STAT_BOX_WIDTH = 400
    R_STAT_BOX_X = WIDTH - R_STAT_BOX_WIDTH  # Position the box at right side


    DEVICES_DEFAULT_LOCATION_START_Y = 10
    DEVICES_DEFAULT_LOCATION_START_X = 10

    AGENTS_DEFAULT_LOCATION_START_Y = 70
    AGENTS_DEFAULT_LOCATION_START_X = 10
    
    LOCATIONS_LEFT_START_Y = HEIGHT / 6 # 768/6 = 128
    LOCATIONS_LEFT_START_X = WIDTH / 8 # 1024/8 = 128

    LOCATIONS_RIGHT_START_Y = LOCATIONS_LEFT_START_Y
    LOCATIONS_RIGHT_START_X = WIDTH - 5 * LOCATIONS_LEFT_START_X

    LOCATIONS_SPAN_X = LOCATIONS_RIGHT_START_X - LOCATIONS_LEFT_START_X

    DOOR_START_X = LOCATIONS_RIGHT_START_X + LOCATIONS_LEFT_START_X
    DOOR_START_Y = LOCATIONS_RIGHT_START_Y + LOCATIONS_SPAN_X / 3

    TRUCK_START_X = DOOR_START_X + LOCATIONS_LEFT_START_X
    TRUCK_START_Y = LOCATIONS_RIGHT_START_Y + LOCATIONS_SPAN_X / 4

    WORKER_WIDTH = 20
    WORKER_HEIGHT = 40

    ROBOT_WIDTH = 40
    ROBOT_HEIGHT = 40

    SPACE_BETWEEN_AGENTS = 5

    FORKLIFT_WIDTH = 40
    FORKLIFT_HEIGHT = 40

    PALLET_JACK_WIDTH = 40
    PALLET_JACK_HEIGHT = 40

    LOCATION_WIDTH = 50
    LOCATION_HEIGHT = 50

    DOOR_WIDTH = 60
    DOOR_HEIGHT = 80

    TRUCK_WIDTH = 80
    TRUCK_HEIGHT = 70

    STEP_DURATION = 0

    # colors
    GRAY = (128, 128, 128)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    # dqn policy agents
    DQN_AGENTS = []


    def encode_tasks_and_devices(self, tasks, devices):
        # encode the tasks df
        type_encoding = {"pick": self.PICK, "put": self.PUT, "load": self.LOAD, "repl": self.REPL} # encode "type"
        tasks["type"] = tasks["type"].map(type_encoding)

        tasks["product"] = tasks["product"].apply(lambda prod:  prod[1:]) # encode product (remove the first character and get the number)
        
        tasks["from loc"] = tasks["from loc"].apply(lambda loc:  loc[1:]) # encode from loc (remove the first character and get the number)
        
        tasks["to loc"] = tasks["to loc"].apply(lambda loc:  loc[1:]) # encode to loc (remove the first character and get the number)
        
        tasks.fillna({'order':'o-1'}, inplace=True)  # encode to order (fill null values by o1, because in next line we are removing the first character)
        tasks["order"] = tasks["order"].apply(lambda order:  order[1:]) # encode to order (remove the first character and get the number)
        
        status_encoding = {"available": self.AVAILABLE, "active": self.ACTIVE, "done": self.DONE} # encode status
        tasks["status"] = tasks["status"].map(status_encoding)

        task_list = tasks.to_numpy().tolist()
        task_list = [list( map(int,i) ) for i in task_list]

        # encode devices
        type_encodings = {"forklift": self.FORKLIFT,"pallet_jack": self.PALLET_JACK, "not_a_device": self.NOT_A_DEVICE} # type encodings
        devices["type"] = devices["type"].map(type_encodings)
        
        status_encoding = {"available": self.AVAILABLE, "active": self.ACTIVE} # encode status
        devices["status"] = devices["status"].map(status_encoding)

        device_list = devices.to_numpy().tolist()
        device_list = [list( map(int,i) ) for i in device_list]

        print("[encoding]: encoded task_list: ", task_list)
        print("[encoding]: encoded device_list: ", device_list)

        return task_list, device_list


   
    def __init__(self):
        super(WarehouseEnv, self).__init__()

        self.time_step = 0 # starting time step of the env
        # make sure to update this value from your main file (if you need to track it in the env too)

        # The action of a agent is selecting a task_id to do.
        # So, action state can be a large number. That means the number of available tasks 
        # can be anything which will be provided in tasks list. 

        # initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        # read the tasks, devices, and locations from csv files
        tasks_df = pd.read_csv('tasks.csv')
        devices_df = pd.read_csv('devices.csv')
        locations_df = pd.read_csv('locations.csv')


        # encode the tasks and devices
        task_list, device_list = self.encode_tasks_and_devices(tasks_df, devices_df)

        # following are the observations that an agent will see from the env at the begining.
        self.tasks = task_list # an observation
        self.devices = device_list # an observation

        self.action_space = spaces.Discrete(len(tasks_df)); # assuming at a given time max number of actions is 100.

        ####################################
        #### Rendering Related Details #####
        ####################################
        # For rendering, its easier maintain live warehoue activities, agent and device default locations in data frames like below
        agents = pd.read_csv('agents.csv')
        
        agent_type_encodings = {"human": self.HUMAN, "robot": self.ROBOT}
        agents["type"] = agents["type"].map(agent_type_encodings)

        agent_status_encoding = {"available": self.AGENT_AVAILABLE, "active": self.AGENT_ACTIVE}
        agents["status"] = agents["status"].map(agent_status_encoding)

        self.agents = agents

        # calculate agent default locations
        agent_locations = pd.DataFrame(columns=['agent_id', 'type', 'default_loc_X', 'default_loc_Y', 'top_left_X', 'top_left_Y'])
        for index, agent in agents.iterrows():
            agent_id = agent['agent_id']
            agent_type = agent['type']

            top_left_X = self.AGENTS_DEFAULT_LOCATION_START_X + \
            (index * (self.WORKER_WIDTH + self.SPACE_BETWEEN_AGENTS) if agent['type'] == self.HUMAN \
             else (index * (self.ROBOT_WIDTH + self.SPACE_BETWEEN_AGENTS)))
            
            top_left_Y = self.AGENTS_DEFAULT_LOCATION_START_Y

            # construct a df and concat it with agent_locations
            # at the begining, the default location and the top_left location are the same
            locations_rec = pd.DataFrame([[agent_id, agent_type, top_left_X, top_left_Y, top_left_X, top_left_Y]], columns=agent_locations.columns)
            agent_locations = pd.concat([agent_locations, locations_rec], ignore_index=True)

        self.agent_locations = agent_locations

        # calculate device default locations
        device_locations = pd.DataFrame(columns=['device_id', 'type', 'default_loc_X', 'default_loc_Y', 'top_left_X', 'top_left_Y'])
        for index, device in devices_df.iterrows():
            device_id = device['device_id']
            device_type = device['type']

            top_left_X = self.DEVICES_DEFAULT_LOCATION_START_X + \
            (index * (self.FORKLIFT_WIDTH + self.SPACE_BETWEEN_AGENTS) if device['type'] == self.FORKLIFT \
             else (index * (self.PALLET_JACK_WIDTH + self.SPACE_BETWEEN_AGENTS)))
            
            top_left_Y = self.DEVICES_DEFAULT_LOCATION_START_Y

            # construct a df and concat it with device_locations
            # at the begining, the default location and the top_left location are the same
            locations_rec = pd.DataFrame([[device_id, device_type, top_left_X, top_left_Y, top_left_X, top_left_Y]], columns=device_locations.columns)
            device_locations = pd.concat([device_locations, locations_rec], ignore_index=True)
        
        self.device_locations = device_locations
        
        # The live locations of agents and devices
        warehouse_now = pd.DataFrame(columns=['agent_id', 'task_id', 'device_id', 'agent_loc', 'device_loc'])
        for index, agent in agents.iterrows():
            agent_id = agent['agent_id']
            task_id = None
            device_id = None
            agent_loc = agent_locations[agent_locations['agent_id'] == agent_id]
            device_loc = None
            
            # construct a df and concat it with warehouse_now
            locations_rec = pd.DataFrame([[agent_id, task_id, device_id, agent_loc, device_loc]], columns=warehouse_now.columns)
            warehouse_now = pd.concat([warehouse_now, locations_rec], ignore_index=True)
        
        self.warehoue_now = warehouse_now 

        # getting all different shelf locations based on locations.csv
        # locations.csv columns are location,type,product,qty
        shelf_locations = pd.DataFrame(columns=['location_id', 'X', 'Y', 'left_or_right', 'type', 'product', 'qty'])

        processed_left_shelves = 0
        processed_right_shelves = 0
        processed_doors = 0
        processed_trucks = 0

        for index, location in locations_df.iterrows():
            location_id = location['location'][1:]
            location_type = location['type']
            product = location['product']
            qty = location['qty']

            if(location_type == 'R' or location_type == 'H'):
                # R = Reserve Shelf (put) location
                # H = Home Shelf (pick) location
                # Note: Replenishments are from R to H
                # if location id (after removing first character) is even, the it is a right shelf
                left_or_right = 'right' if int(location_id) % 2 == 0 else 'left'
            
                if int(location_id) % 2 == 0:
                    x = self.LOCATIONS_RIGHT_START_X
                    y = self.LOCATIONS_RIGHT_START_Y + (processed_right_shelves * self.LOCATION_HEIGHT)
                    processed_right_shelves += 1
                else:
                    x = self.LOCATIONS_LEFT_START_X
                    y = self.LOCATIONS_LEFT_START_Y + (processed_left_shelves * self.LOCATION_HEIGHT)
                    processed_left_shelves += 1
            elif(location_type == 'D'):
                left_or_right = 'none'
                x = self.DOOR_START_X 
                y = self.DOOR_START_Y + (processed_doors * self.DOOR_HEIGHT + 10)
                processed_doors += 1
            elif(location_type == 'T'):
                left_or_right = 'none'
                x = self.TRUCK_START_X
                y = self.TRUCK_START_Y + (processed_trucks * self.TRUCK_HEIGHT + 10)
                processed_trucks += 1

            # construct a df and concat it with shelf_locations
            locations_rec = pd.DataFrame([[location_id, x, y, left_or_right, location_type, product, qty]], \
                                         columns=shelf_locations.columns)
            shelf_locations = pd.concat([shelf_locations, locations_rec], ignore_index=True)

        
        print("==>[from env - __init__]: shelf_locations: \n", shelf_locations)
        self.shelf_locations = shelf_locations    

        ####################################
        ####      EPISODE REPORTS      #####
        #################################### 
        # A dataframe to store the task started time, who did it and when it is done
        # this will be used to create a report and visualize the performance of the agents
        task_report = pd.DataFrame(columns=["task_id", "agent_id", "device_id", "task_time", "start_time", "end_time"])
        self.task_report = task_report
        

        ####################################
        ####     OBSERVATION SPACE     #####
        ####################################
        # OBSERVATIONS FROM ENV ARE: warehouse tasks, devices
        # Warehouse Tasks
        self.task_shape = (len(self.tasks), 8)  # Adjust if you have more task attributes
        self.task_low = np.array([[
            0,  # Task_Id (assume non-negative integers)
            0,  # Type (categorical)
            0,  # Product
            0,  # Qty
            0,  # From Location
            0,  # To Location
            0,  # Time (might be a float)
            0,  # Order No
            0   # Status (categorical)
        ] for _ in range(len(tasks_df))])
        self.task_high = np.array([[
            100,  # Task_Id, maximum task id
            3,  # Task Type (if categorical, maximum category value). 3 since we start from 0. so there are four types
            100,  # Product
            100,  # Qty
            100,  # From Location 
            100,  # To Location
            60,   # Time (if assuming a 24-hour window) 
            100,  # Order No
            2   # Status  (0 - available, 1 - active, 2 - done)
        ] for _ in range(len(tasks_df))])

        # Warehouse Devices
        self.device_shape = (len(self.devices),3)  # Adjust if you have more device attributes
        self.device_low = np.array([[
            0,  # Device_Id
            0,  # Type 
            0,   # Status
            0, # current_task_id
        ] for _ in range(len(devices_df))])
        self.device_high = np.array([[
            100,  # Device_Id
            2,  # Type (0-pallet_jack, 1-forklift, 2-not_a_device)
            1,  # Status (0-available, 1-active)
            999 # current_task_id (999 means no task assigned)
        ] for _ in range(len(devices_df))])

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
        self.time_step = 0 # starting time step of the env
        # make sure to update this value from your main file (if you need to track it in the env too)

        # The action of a agent is selecting a task_id to do.
        # So, action state can be a large number. That means the number of available tasks 
        # can be anything which will be provided in tasks list. 

        # initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        # read the tasks and devices from csv files
        tasks_df = pd.read_csv('tasks.csv')
        devices_df = pd.read_csv('devices.csv')


        # encode the tasks and devices
        task_list, device_list = self.encode_tasks_and_devices(tasks_df, devices_df)

        # following are the observations that an agent will see from the env at the begining.
        self.tasks = task_list # an observation
        self.devices = device_list # an observation

        self.action_space = spaces.Discrete(len(tasks_df)); # assuming at a given time max number of actions is 100.

        ####################################
        #### Rendering Related Details #####
        ####################################
        # For rendering, its easier maintain live warehoue activities, agent and device default locations in data frames like below
        agents = pd.read_csv('agents.csv')
        
        agent_type_encodings = {"human": self.HUMAN, "robot": self.ROBOT}
        agents["type"] = agents["type"].map(agent_type_encodings)

        agent_status_encoding = {"available": self.AGENT_AVAILABLE, "active": self.AGENT_ACTIVE}
        agents["status"] = agents["status"].map(agent_status_encoding)

        self.agents = agents

        # calculate agent default locations
        agent_locations = pd.DataFrame(columns=['agent_id', 'type', 'default_loc_X', 'default_loc_Y', 'top_left_X', 'top_left_Y'])
        for index, agent in agents.iterrows():
            agent_id = agent['agent_id']
            agent_type = agent['type']

            top_left_X = self.AGENTS_DEFAULT_LOCATION_START_X + \
            (index * (self.WORKER_WIDTH + self.SPACE_BETWEEN_AGENTS) if agent['type'] == self.HUMAN \
             else (index * (self.ROBOT_WIDTH + self.SPACE_BETWEEN_AGENTS)))
            
            top_left_Y = self.AGENTS_DEFAULT_LOCATION_START_Y

            # construct a df and concat it with agent_locations
            # at the begining, the default location and the top_left location are the same
            locations_rec = pd.DataFrame([[agent_id, agent_type, top_left_X, top_left_Y, top_left_X, top_left_Y]], columns=agent_locations.columns)
            agent_locations = pd.concat([agent_locations, locations_rec], ignore_index=True)

        self.agent_locations = agent_locations

        # calculate device default locations
        device_locations = pd.DataFrame(columns=['device_id', 'type', 'default_loc_X', 'default_loc_Y', 'top_left_X', 'top_left_Y'])
        for index, device in devices_df.iterrows():
            device_id = device['device_id']
            device_type = device['type']

            top_left_X = self.DEVICES_DEFAULT_LOCATION_START_X + \
            (index * (self.FORKLIFT_WIDTH + self.SPACE_BETWEEN_AGENTS) if device['type'] == self.FORKLIFT \
             else (index * (self.PALLET_JACK_WIDTH + self.SPACE_BETWEEN_AGENTS)))
            
            top_left_Y = self.DEVICES_DEFAULT_LOCATION_START_Y

            # construct a df and concat it with device_locations
            # at the begining, the default location and the top_left location are the same
            locations_rec = pd.DataFrame([[device_id, device_type, top_left_X, top_left_Y, top_left_X, top_left_Y]], columns=device_locations.columns)
            device_locations = pd.concat([device_locations, locations_rec], ignore_index=True)
        
        self.device_locations = device_locations
        
        # The live locations of agents and devices
        warehouse_now = pd.DataFrame(columns=['agent_id', 'task_id', 'device_id', 'agent_loc', 'device_loc'])
        for index, agent in agents.iterrows():
            agent_id = agent['agent_id']
            task_id = None
            device_id = None
            agent_loc = agent_locations[agent_locations['agent_id'] == agent_id]
            device_loc = None
            
            # construct a df and concat it with warehouse_now
            locations_rec = pd.DataFrame([[agent_id, task_id, device_id, agent_loc, device_loc]], columns=warehouse_now.columns)
            warehouse_now = pd.concat([warehouse_now, locations_rec], ignore_index=True)
        
        self.warehoue_now = warehouse_now 

        # getting all different shelf locations based on provided task list (tasks.csv)
        locations = [task[self.TASK_FROM_LOC] for task in self.tasks]
        locations.extend([task[self.TASK_TO_LOC] for task in self.tasks])
        locations = list(set(locations)) # remove duplicates and get unique locations
        locations.sort() # sort the locations
        self.locations = locations

        # left locations are odd numbers and right locations are even numbers
        locations_left = [loc for loc in locations if loc % 2 != 0]
        locations_right = [loc for loc in locations if loc % 2 == 0]
        self.locations_left = locations_left
        self.locations_right = locations_right

        # shelf locations
        shelf_locations = pd.DataFrame(columns=['location_id', 'X', 'Y', 'left_or_right'])
        for index, loc in enumerate(self.locations_left):
            x = self.LOCATIONS_LEFT_START_X
            y = self.LOCATIONS_LEFT_START_Y + (index * self.LOCATION_HEIGHT)

            self_location_rec = pd.DataFrame([[loc, x, y, 'left']], columns=shelf_locations.columns)
            shelf_locations = pd.concat([shelf_locations, self_location_rec], ignore_index=True)

        for index, loc in enumerate(self.locations_right):
            x = self.LOCATIONS_RIGHT_START_X
            y = self.LOCATIONS_RIGHT_START_Y + (index * self.LOCATION_HEIGHT)

            self_location_rec = pd.DataFrame([[loc, x, y, 'right']], columns=shelf_locations.columns)
            shelf_locations = pd.concat([shelf_locations, self_location_rec], ignore_index=True)
        
        print("==>[from env - __init__]: shelf_locations: \n", shelf_locations)
        self.shelf_locations = shelf_locations    

        ####################################
        ####      EPISODE REPORTS      #####
        #################################### 
        # A dataframe to store the task started time, who did it and when it is done
        # this will be used to create a report and visualize the performance of the agents
        task_report = pd.DataFrame(columns=["task_id", "agent_id", "device_id", "task_time", "start_time", "end_time"])
        self.task_report = task_report
        

        ####################################
        ####     OBSERVATION SPACE     #####
        ####################################
        # OBSERVATIONS FROM ENV ARE: warehouse tasks, devices
        # Warehouse Tasks
        self.task_shape = (len(self.tasks), 8)  # Adjust if you have more task attributes
        self.task_low = np.array([[
            0,  # Task_Id (assume non-negative integers)
            0,  # Type (categorical)
            0,  # Product
            0,  # From Location
            0,  # To Location
            0,  # Time (might be a float)
            0,  # Order No
            0   # Status (categorical)
        ] for _ in range(len(tasks_df))])
        self.task_high = np.array([[
            100,  # Task_Id, maximum task id
            3,  # Task Type (if categorical, maximum category value). 3 since we start from 0. so there are four types
            100,  # Product
            100,  # From Location 
            100,  # To Location
            60,   # Time (if assuming a 24-hour window) 
            100,  # Order No
            2   # Status  (0 - available, 1 - active, 2 - done)
        ] for _ in range(len(tasks_df))])

        # Warehouse Devices
        self.device_shape = (len(self.devices),3)  # Adjust if you have more device attributes
        self.device_low = np.array([[
            0,  # Device_Id
            0,  # Type 
            0,   # Status
            0, # current_task_id
        ] for _ in range(len(devices_df))])
        self.device_high = np.array([[
            100,  # Device_Id
            2,  # Type (0-pallet_jack, 1-forklift, 2-not_a_device)
            1,  # Status (0-available, 1-active)
            999 # current_task_id (999 means no task assigned)
        ] for _ in range(len(devices_df))])

        # print(self.task_low)

        # Combining spaces using dictionaries
        self.observation_space = spaces.Dict({
            'tasks': spaces.Box(low=self.task_low, high=self.task_high, shape=np.array(self.tasks).shape, dtype=np.int32),
            'devices': spaces.Box(low=self.device_low, high=self.device_high, shape=np.array(self.devices).shape, dtype=np.int32),
        })


        info = {} # additional information

        return (self.observation_space, info)


    def step(self, agent, action, time_step=0):
        action_index = action - 1
        agent_type = self.agents.query(f'agent_id == {agent}')['type'].values[0]
        agent_type_str = "human" if agent_type == self.HUMAN else "robot" 
        agent_status = self.agents.query(f'agent_id == {agent}')['status'].values[0] 
        agent_current_task = self.agents.query(f'agent_id == {agent}')['current_task'].values[0]
        agent_current_task_start_time = self.agents.query(f'agent_id == {agent}')['current_task_start_time'].values[0]
        agent_current_device = self.agents.query(f'agent_id == {agent}')['current_device'].values[0]
        agent_current_reward = self.agents.query(f'agent_id == {agent}')['reward'].values[0]
        reward = 0 # step reward

        # Here, an action will be a task_id and based on task status we can return a reward.
        print(f"=======> [from env - step]: {agent_type_str} agent with id: {agent} chose action (task id): {action} with device: {agent_current_device} at Time Step: {time_step}")


        # check if agent is still active (doing a task)
        if agent_status == self.ACTIVE:

            # if agent is still active and have more time to complete the task, move him towards to_loc
            if time_step <= agent_current_task_start_time + self.tasks[agent_current_task - 1][self.TASK_TIME]:
                print(f"=======> [from env - step]: agent is still active doing task {agent_current_task}, hence skipping to next agent")
                
                # update agent location (decide based on current time step, task from location and task to location)
                # also move (update) the selected device.
                task_duration = self.tasks[agent_current_task - 1][self.TASK_TIME]
                task_from_loc = self.tasks[agent_current_task - 1][self.TASK_FROM_LOC]
                print("task_from_loc: ", task_from_loc)
                # print(self.shelf_locations[self.shelf_locations['location_id'] == '22']['X'].values[0])
                task_from_loc_X = self.shelf_locations[self.shelf_locations['location_id'] == str(task_from_loc)]['X'].values[0]
                
                # if task_from_loc is a left location, make X coordinate of the agent a little right to the shelf
                if self.shelf_locations[self.shelf_locations['location_id'] == str(task_from_loc)]['left_or_right'].values[0] == 'left':
                    task_from_loc_X = task_from_loc_X + self.LOCATION_WIDTH
                else:
                    task_from_loc_X = task_from_loc_X - 20
                task_from_loc_Y = self.shelf_locations[self.shelf_locations['location_id'] == str(task_from_loc)]['Y'].values[0]

                task_to_loc = self.tasks[agent_current_task - 1][self.TASK_TO_LOC]
                task_to_loc_X = self.shelf_locations[self.shelf_locations['location_id'] == str(task_to_loc)]['X'].values[0]
                
                # if task_to_loc is a left location, make X coordinate of the agent destination a little right to the shelf
                if self.shelf_locations[self.shelf_locations['location_id'] == str(task_to_loc)]['left_or_right'].values[0] == 'left':
                    task_to_loc_X = task_to_loc_X + self.LOCATION_WIDTH
                else:
                    task_to_loc_X = task_to_loc_X - 20
                task_to_loc_Y = self.shelf_locations[self.shelf_locations['location_id'] == str(task_to_loc)]['Y'].values[0]

                # distance to be covered by the agent in each time step
                X_distance_per_time_step = (task_to_loc_X - task_from_loc_X) / task_duration
                Y_distance_per_time_step = (task_to_loc_Y - task_from_loc_Y) / task_duration

                from_location_l_or_r = self.shelf_locations[self.shelf_locations['location_id'] == str(task_from_loc)]['left_or_right'].values[0]

                to_location_type = self.shelf_locations[self.shelf_locations['location_id'] == str(task_to_loc)]['type'].values[0]
                from_location_type = self.shelf_locations[self.shelf_locations['location_id'] == str(task_from_loc)]['type'].values[0]
                
                # Hence based on agent_current_task_start_time, time_step, task_duration, diagonal_distance, 
                # task_from_loc_X, task_from_loc_Y, task_to_loc_X, task_to_loc_Y, X_distance_per_time_step and
                # Y_distance_per_time_step, the X coordinate and Y coordinate of the agent at this moment is:
                if from_location_l_or_r == 'left' and task_to_loc_X > task_from_loc_X:
                    # agent should go from left to right
                    agent_X = task_from_loc_X + (X_distance_per_time_step * (time_step - agent_current_task_start_time))
                    agent_Y = task_from_loc_Y + (Y_distance_per_time_step * (time_step - agent_current_task_start_time))
                elif from_location_l_or_r == 'right' and task_to_loc_X < task_from_loc_X:
                    # agent should go from right to left
                    # here X_distance_per_time_step is negative
                    agent_X = task_from_loc_X + (X_distance_per_time_step * (time_step - agent_current_task_start_time))
                    agent_Y = task_from_loc_Y + (Y_distance_per_time_step * (time_step - agent_current_task_start_time))
                elif to_location_type == 'D' or from_location_type == 'D':
                    # agent should go to or come from door
                    agent_X = task_from_loc_X + (X_distance_per_time_step * (time_step - agent_current_task_start_time))
                    agent_Y = task_from_loc_Y + (Y_distance_per_time_step * (time_step - agent_current_task_start_time))
                elif to_location_type == 'T' or from_location_type == 'T':
                    # agent should go to or come from truck
                    agent_X = task_from_loc_X + (X_distance_per_time_step * (time_step - agent_current_task_start_time))
                    agent_Y = task_from_loc_Y + (Y_distance_per_time_step * (time_step - agent_current_task_start_time))
                elif int(X_distance_per_time_step) == 0 and Y_distance_per_time_step < 0:
                    # agent should go up
                    agent_X = task_from_loc_X
                    agent_Y = task_from_loc_Y + (Y_distance_per_time_step * (time_step - agent_current_task_start_time))
                    # here Y_distance_per_time_step is negative
                elif int(X_distance_per_time_step) == 0 and Y_distance_per_time_step > 0:
                    # agent should go down
                    agent_X = task_from_loc_X
                    agent_Y = task_from_loc_Y + (Y_distance_per_time_step * (time_step - agent_current_task_start_time))
                    # here Y_distance_per_time_step is positive
                

                self.agent_locations.loc[self.agent_locations['agent_id'] == agent, 'top_left_X'] = agent_X
                self.agent_locations.loc[self.agent_locations['agent_id'] == agent, 'top_left_Y'] = agent_Y

                # update the device location
                if agent_current_device != -1:
                    device_X = agent_X
                    device_Y = agent_Y
                    self.device_locations.loc[self.device_locations['device_id'] == agent_current_device, 'top_left_X'] = device_X
                    self.device_locations.loc[self.device_locations['device_id'] == agent_current_device, 'top_left_Y'] = device_Y

                
                
                reward = 0
            else: # if agent is done with the task, make agent free
                print(f"=======> [from env - step]: agent is finishing the task {agent_current_task} at time step {time_step}")

                # update task report
                self.task_report.loc[self.task_report["task_id"] == agent_current_task, "end_time"] = time_step

                # make the agent free
                self.agents.loc[self.agents['agent_id'] == agent, "status"] = self.AGENT_AVAILABLE
                self.agents.loc[self.agents['agent_id'] == agent, 'current_task'] = -1
                self.agents.loc[self.agents['agent_id'] == agent, 'current_device'] = -1
                self.agents.loc[self.agents['agent_id'] == agent, 'current_task_start_time'] = -1

                # move agent to default location
                self.agent_locations.loc[self.agent_locations['agent_id'] == agent, 'top_left_X'] = self.agent_locations.query(f'agent_id == {agent}')['default_loc_X'].values[0]
                self.agent_locations.loc[self.agent_locations['agent_id'] == agent, 'top_left_Y'] = self.agent_locations.query(f'agent_id == {agent}')['default_loc_Y'].values[0]

                # move the device to default location
                if agent_current_device != -1:
                    self.device_locations.loc[self.device_locations['device_id'] == agent_current_device, 'top_left_X'] = self.device_locations.query(f'device_id == {agent_current_device}')['default_loc_X'].values[0]
                    self.device_locations.loc[self.device_locations['device_id'] == agent_current_device, 'top_left_Y'] = self.device_locations.query(f'device_id == {agent_current_device}')['default_loc_Y'].values[0]
                
                # make the task done
                self.tasks[agent_current_task - 1][self.TASK_STATUS] = self.DONE

                # free the device
                self.devices[agent_current_device - 1][self.DEVICE_STATUS] = self.AVAILABLE
                self.devices[agent_current_device - 1][self.DEVICE_CURRENT_TASK_ID] = -1

                print(f"=======> [from env - step]: agent is now in {self.AGENT_AVAILABLE} state at time step {time_step}")
        else: # If the agent is Available (FREE to take up the suggested task)
            # REWARD CALCULATION
            if action > len(self.tasks):
                # Punish if the selected task is grater than available number of tasks
                print(f"=======> [from env]: Received a task id {action} is grater than available number of tasks")
                reward = -1
                # update the reward of the agent
                self.agents.loc[self.agents['agent_id'] == agent, "reward"] = agent_current_reward + reward
                
            elif self.tasks[action_index][self.TASK_STATUS] != self.AVAILABLE:
                # Punish if the selected task is alreay active (Assigned) or completed (done)
                print(f"=======> [from env - step]: selected task {action} is not available. It is active or already done!")
                reward = -1
                # update the reward of the agent
                self.agents.loc[self.agents['agent_id'] == agent, "reward"] = agent_current_reward + reward
            else:
                # make agent active and assign the task
                self.agents.loc[self.agents['agent_id'] == agent, "status"] = self.AGENT_ACTIVE
                self.agents.loc[self.agents['agent_id'] == agent, 'current_task'] = action
                self.agents.loc[self.agents['agent_id'] == agent, 'current_task_start_time'] = time_step

                # if agent is human
                if agent_type == self.HUMAN:

                    print("=======> [from env - step]: in human reward calculation 1")

                    # if task is a pick, a pallet_jack is needed. 
                    if self.tasks[action_index][self.TASK_TYPE] == self.PICK:
                        print("=======> [from env - step]: task is a pick task")
                        pallet_jacks = [device for device in self.devices if device[self.DEVICE_TYPE] == self.PALLET_JACK]
                        num_available_pallet_jacks = len([pj for pj in pallet_jacks if pj[self.DEVICE_STATUS] == self.AVAILABLE])

                        print("=======> [from env - step]: number of available pallet jacks: ", num_available_pallet_jacks)

                        if num_available_pallet_jacks > 0:
                            reward = 1
                            # update the reward of the agent
                            self.agents.loc[self.agents['agent_id'] == agent, "reward"] = agent_current_reward + reward

                            # make the task active
                            self.tasks[action_index][self.TASK_STATUS] = self.ACTIVE

                            # make that pallet jack status as active (the first pj which is available)
                            for index, device in enumerate(self.devices):
                                if device[self.DEVICE_TYPE] == self.PALLET_JACK and device[self.DEVICE_STATUS] == self.AVAILABLE:
                                    self.devices[index][self.DEVICE_STATUS] = self.ACTIVE
                                    self.devices[index][self.DEVICE_CURRENT_TASK_ID] = action
                                    # assign the device to agent too
                                    self.agents.loc[self.agents['agent_id'] == agent, 'current_device'] = device[self.DEVICE_ID]
                                    break
                        else:
                            # as there is no available device to work on this task, free the agent and move to default location
                            self.agents.loc[self.agents['agent_id'] == agent, "status"] = self.AGENT_AVAILABLE
                            self.agents.loc[self.agents['agent_id'] == agent, 'current_task'] = -1
                            self.agents.loc[self.agents['agent_id'] == agent, 'current_device'] = -1
                            self.agents.loc[self.agents['agent_id'] == agent, 'current_task_start_time'] = -1

                            # move agent to default location
                            self.agent_locations.loc[self.agent_locations['agent_id'] == agent, 'top_left_X'] = self.agent_locations.query(f'agent_id == {agent}')['default_loc_X'].values[0]
                            self.agent_locations.loc[self.agent_locations['agent_id'] == agent, 'top_left_Y'] = self.agent_locations.query(f'agent_id == {agent}')['default_loc_Y'].values[0]

                            reward = -1 # punish if no available pallet jacks
                            # update the reward of the agent
                            self.agents.loc[self.agents['agent_id'] == agent, "reward"] = agent_current_reward + reward
                            # Agent should learn to not to select tasks if no available devices (although there are available tasks)
                        
                    else: #every other task types needs a forklift 
                        print("=======> [from env - step]: task is a forklift task")
                        forklifts = [device for device in self.devices if device[self.DEVICE_TYPE] == self.FORKLIFT]
                        num_available_forklifts = len([fk for fk in forklifts if fk[self.DEVICE_STATUS] == self.AVAILABLE])

                        print("=======> [from env - step]: number of available forklifts: ", num_available_forklifts)

                        if num_available_forklifts > 0:
                            reward = 1
                            
                            # update the reward of the agent
                            self.agents.loc[self.agents['agent_id'] == agent, "reward"] = agent_current_reward + reward

                            # make the task active
                            self.tasks[action_index][self.TASK_STATUS] = self.ACTIVE

                            # make that forklift status as active (the first fk which is available)
                            for index, device in enumerate(self.devices):
                                if device[self.DEVICE_TYPE] == self.FORKLIFT and device[self.DEVICE_STATUS] == self.AVAILABLE:
                                    self.devices[index][self.DEVICE_STATUS] = self.ACTIVE
                                    self.devices[index][self.DEVICE_CURRENT_TASK_ID] = action
                                    # assign the device to agent too
                                    self.agents.loc[self.agents['agent_id'] == agent, 'current_device'] = device[self.DEVICE_ID]
                                    break
                        else:
                            # as there is no available device to work on this task, free the agent and move to default location
                            self.agents.loc[self.agents['agent_id'] == agent, "status"] = self.AGENT_AVAILABLE
                            self.agents.loc[self.agents['agent_id'] == agent, 'current_task'] = -1
                            self.agents.loc[self.agents['agent_id'] == agent, 'current_device'] = -1
                            self.agents.loc[self.agents['agent_id'] == agent, 'current_task_start_time'] = -1

                            # move agent to default location
                            self.agent_locations.loc[self.agent_locations['agent_id'] == agent, 'top_left_X'] = self.agent_locations.query(f'agent_id == {agent}')['default_loc_X'].values[0]
                            self.agent_locations.loc[self.agent_locations['agent_id'] == agent, 'top_left_Y'] = self.agent_locations.query(f'agent_id == {agent}')['default_loc_Y'].values[0]
                            
                            reward = -1 # punish if no available pallet jacks

                            # update the reward of the agent
                            self.agents.loc[self.agents['agent_id'] == agent, "reward"] = agent_current_reward + reward

                            # This is because the agent should learn to not to select tasks if no available devices 
                            # (although there are available tasks)
                        print("=======> [from env - step]: reward: ", reward)

                else: # robots can only do pick tasks and they dont need any devices for that. 

                    if self.tasks[action_index][self.TASK_TYPE] != self.PICK: # if the task type is not a pick task, punish
                        
                        # as the robot selected task is wrong make it free
                        self.agents.loc[self.agents['agent_id'] == agent, "status"] = self.AGENT_AVAILABLE
                        self.agents.loc[self.agents['agent_id'] == agent, 'current_task'] = -1
                        self.agents.loc[self.agents['agent_id'] == agent, 'current_device'] = -1
                        self.agents.loc[self.agents['agent_id'] == agent, 'current_task_start_time'] = -1

                        # move agent to default location
                        self.agent_locations.loc[self.agent_locations['agent_id'] == agent, 'top_left_X'] = self.agent_locations.query(f'agent_id == {agent}')['default_loc_X'].values[0]
                        self.agent_locations.loc[self.agent_locations['agent_id'] == agent, 'top_left_Y'] = self.agent_locations.query(f'agent_id == {agent}')['default_loc_Y'].values[0]


                        reward = -1
                        # update the reward of the agent
                        self.agents.loc[self.agents['agent_id'] == agent, "reward"] = agent_current_reward + reward
                    else: 
                        reward = 1
                        # update the reward of the agent
                        self.agents.loc[self.agents['agent_id'] == agent, "reward"] = agent_current_reward + reward
                        
                        # make the task to active status
                        self.tasks[action_index][self.TASK_STATUS] = self.ACTIVE

            # add a record to task report
            agent_current_device = self.agents.query(f'agent_id == {agent}')['current_device'].values[0]
            # if only the reward is postive add the record (reward negative means there was a problem getting the task)
            if reward > 0:
                task_report_rec = pd.DataFrame([[action, agent, agent_current_device, self.tasks[action_index][self.TASK_TIME], time_step, None]], columns=self.task_report.columns)
                self.task_report = pd.concat([self.task_report, task_report_rec], ignore_index=True)

        print("=======> [from env - step]: reward: ", reward)
        # DONE?
        num_available_tasks = len([task for task in self.tasks if task[self.TASK_STATUS] == self.AVAILABLE])
        if num_available_tasks > 0:
            done = False
        else:
            done = True

        print("=======> [from env - step]: after this, num_available_tasks: ", num_available_tasks)
        print("=======> [from env - step]: done: ", done)

        # TRUNCATED?
        truncated = False

        print("=======> [from env - step]: truncated: ", truncated)
        

        # INFO?
        info = {}

        observation = {
            "tasks": self.tasks,
            "devices": self.devices
        }

        # render the pygame
        self.render()

        return observation, reward, done, truncated, info
    
    # render locations (shelves)
    def _render_locations(self):
        location_left = pygame.image.load("location_left.png")
        location_left = pygame.transform.scale(location_left, (self.LOCATION_WIDTH, self.LOCATION_HEIGHT))
        location_left_rect = location_left.get_rect()

        location_right = pygame.image.load("location_right.png")
        location_right = pygame.transform.scale(location_right, (self.LOCATION_WIDTH, self.LOCATION_HEIGHT))
        location_right_rect = location_right.get_rect()

        door = pygame.image.load("door.png")
        door = pygame.transform.scale(door, (self.DOOR_WIDTH, self.DOOR_HEIGHT))
        door_rect = door.get_rect()

        truck = pygame.image.load("truck.png")
        truck = pygame.transform.scale(truck, (self.TRUCK_WIDTH, self.TRUCK_HEIGHT))
        truck_rect = truck.get_rect()

        for index, loc in self.shelf_locations.iterrows():
            if loc['left_or_right'] == 'left':
                location_left_rect.x = loc['X']
                location_left_rect.y = loc['Y']
                self.screen.blit(location_left, location_left_rect)
            elif loc['left_or_right'] == 'right':
                location_right_rect.x = loc['X']
                location_right_rect.y = loc['Y']
                self.screen.blit(location_right, location_right_rect)
            elif loc['type'] == 'D':
                door_rect.x = loc['X']
                door_rect.y = loc['Y']
                self.screen.blit(door, door_rect)
            elif loc['type'] == 'T':
                truck_rect.x = loc['X']
                truck_rect.y = loc['Y']
                self.screen.blit(truck, truck_rect)

    # render devices
    def _render_devices(self):
        pallet_jack = pygame.image.load("pallet_jack.png")
        pallet_jack = pygame.transform.scale(pallet_jack, (self.PALLET_JACK_WIDTH, self.PALLET_JACK_HEIGHT))
        pallet_jack_rect = pallet_jack.get_rect()

        # render pallet_jacks
        for index, device in self.device_locations.iterrows():
            if device['type'] == self.PALLET_JACK:
                pallet_jack_rect.x = device['top_left_X']
                pallet_jack_rect.y = device['top_left_Y']
                self.screen.blit(pallet_jack, pallet_jack_rect)
        
        forklift = pygame.image.load("forklift.png")
        forklift = pygame.transform.scale(forklift, (self.FORKLIFT_WIDTH, self.FORKLIFT_HEIGHT))
        forklift_rect = forklift.get_rect()

        # render forklifts
        for index, device in self.device_locations.iterrows():
            if device['type'] == self.FORKLIFT:
                forklift_rect.x = device['top_left_X']
                forklift_rect.y = device['top_left_Y']
                self.screen.blit(forklift, forklift_rect)
    
    # render agents
    def _render_agents(self):
        worker = pygame.image.load("warehouse_worker.png")
        worker = pygame.transform.scale(worker, (self.WORKER_WIDTH, self.WORKER_HEIGHT))
        worker_rect = worker.get_rect()

        worker_dqn = pygame.image.load("warehouse_worker_dqn.png")
        worker_dqn = pygame.transform.scale(worker_dqn, (self.WORKER_WIDTH, self.WORKER_HEIGHT))
        worker_dqn_rect = worker_dqn.get_rect()

        robot = pygame.image.load("selector_robot.png")
        robot = pygame.transform.scale(robot, (self.ROBOT_WIDTH, self.ROBOT_HEIGHT))
        robot_rect = robot.get_rect()

        robot_dqn = pygame.image.load("selector_robot_dqn.png")
        robot_dqn = pygame.transform.scale(robot_dqn, (self.ROBOT_WIDTH, self.ROBOT_HEIGHT))
        robot_dqn_rect = robot_dqn.get_rect()

        # render workers
        for index, agent in self.agent_locations.iterrows():
            if agent['type'] == self.HUMAN:
                # if agent is a dqn policy agent, use worker_dqn_rect
                if agent['agent_id'] in self.DQN_AGENTS:
                    worker_dqn_rect.x = agent['top_left_X']
                    worker_dqn_rect.y = agent['top_left_Y']
                    self.screen.blit(worker_dqn, worker_dqn_rect)
                else:
                    worker_rect.x = agent['top_left_X']
                    worker_rect.y = agent['top_left_Y']
                    self.screen.blit(worker, worker_rect)
        
        # render robots
        for index, agent in self.agent_locations.iterrows():
            if agent['type'] == self.ROBOT:
                # if agent is a dqn policy agent, use robot_dqn_rect
                if agent['agent_id'] in self.DQN_AGENTS:
                    robot_dqn_rect.x = agent['top_left_X']
                    robot_dqn_rect.y = agent['top_left_Y']
                    self.screen.blit(robot_dqn, robot_dqn_rect)
                else:
                    robot_rect.x = agent['top_left_X']
                    robot_rect.y = agent['top_left_Y']
                    self.screen.blit(robot, robot_rect)
    
    # render stats
    def _render_stats(self):

        stat_left_column_X = self.R_STAT_BOX_X + 10
        stat_right_column_X = self.R_STAT_BOX_X + 200

        # draw R_STAT_BOX
        pygame.draw.rect(self.screen, self.GRAY, (self.R_STAT_BOX_X, 0, self.R_STAT_BOX_WIDTH, self.HEIGHT))

        # show total task number
        font = pygame.font.Font(None, 28)
        text = font.render(f"Tasks: {len(self.tasks)}", True, self.WHITE)
        self.screen.blit(text, (stat_left_column_X, self.LOCATIONS_RIGHT_START_Y))

        # number of pick tasks
        font = pygame.font.Font(None, 24)
        num_pick_tasks = len([task for task in self.tasks if task[self.TASK_TYPE] == self.PICK])
        text = font.render(f"Pick Tasks: {num_pick_tasks}", True, self.WHITE)
        self.screen.blit(text, (stat_left_column_X, self.LOCATIONS_RIGHT_START_Y + 40))

        # number of repln tasks
        num_repln_tasks = len([task for task in self.tasks if task[self.TASK_TYPE] == self.REPL])
        text = font.render(f"Replenish Tasks: {num_repln_tasks}", True, self.WHITE)
        self.screen.blit(text, (stat_left_column_X, self.LOCATIONS_RIGHT_START_Y + 80))

        # number of put tasks
        num_put_tasks = len([task for task in self.tasks if task[self.TASK_TYPE] == self.PUT])
        text = font.render(f"Put Tasks: {num_put_tasks}", True, self.WHITE)
        self.screen.blit(text, (stat_left_column_X, self.LOCATIONS_RIGHT_START_Y + 120))

        # number of load tasks
        num_load_tasks = len([task for task in self.tasks if task[self.TASK_TYPE] == self.LOAD])
        text = font.render(f"Load Tasks: {num_load_tasks}", True, self.WHITE)
        self.screen.blit(text, (stat_left_column_X, self.LOCATIONS_RIGHT_START_Y + 160))


        # Show Total Devices and Device type counts in right column
        font = pygame.font.Font(None, 28)
        text = font.render(f"Devices: {len(self.devices)}", True, self.WHITE)
        self.screen.blit(text, (stat_right_column_X, self.LOCATIONS_RIGHT_START_Y))

        # number of pallet jacks
        font = pygame.font.Font(None, 24)
        num_pallet_jacks = len([device for device in self.devices if device[self.DEVICE_TYPE] == self.PALLET_JACK])
        text = font.render(f"Pallet Jacks: {num_pallet_jacks}", True, self.WHITE)
        self.screen.blit(text, (stat_right_column_X, self.LOCATIONS_RIGHT_START_Y + 40))

        # number of forklifts
        num_forklifts = len([device for device in self.devices if device[self.DEVICE_TYPE] == self.FORKLIFT])
        text = font.render(f"Forklifts: {num_forklifts}", True, self.WHITE)
        self.screen.blit(text, (stat_right_column_X, self.LOCATIONS_RIGHT_START_Y + 80))

        # Draw a horizontal line at the end of task stats with a width of stat box
        pygame.draw.line(self.screen, self.WHITE, (self.R_STAT_BOX_X, self.LOCATIONS_RIGHT_START_Y + 200), \
                         (self.R_STAT_BOX_X + self.R_STAT_BOX_WIDTH, self.LOCATIONS_RIGHT_START_Y + 200), 2)
        

        # Show Agent Stats and Agent type counts in left column after the above horizontal line
        font = pygame.font.Font(None, 28)
        text = font.render(f"Agents: {len(self.agents)}", True, self.WHITE)
        self.screen.blit(text, (stat_left_column_X, self.LOCATIONS_RIGHT_START_Y + 220))

        # number of human agents
        font = pygame.font.Font(None, 24)
        num_human_agents = len(self.agents.query(f'type == {self.HUMAN}'))
        text = font.render(f"Humans: {num_human_agents}", True, self.WHITE)
        self.screen.blit(text, (stat_left_column_X, self.LOCATIONS_RIGHT_START_Y + 260))

        # number of robot agents
        num_robot_agents = len(self.agents.query(f'type == {self.ROBOT}'))
        text = font.render(f"Robots: {num_robot_agents}", True, self.WHITE)
        self.screen.blit(text, (stat_right_column_X, self.LOCATIONS_RIGHT_START_Y + 260))

        # Draw a horizontal line at the end of agent stats with a width of stat box
        pygame.draw.line(self.screen, self.WHITE, (self.R_STAT_BOX_X, self.LOCATIONS_RIGHT_START_Y + 300), \
                         (self.R_STAT_BOX_X + self.R_STAT_BOX_WIDTH, self.LOCATIONS_RIGHT_START_Y + 300), 2)
        

        # Show Dynamic Statas like number of reamining (available) tasks to do, number of free devices, number of free agents
        font = pygame.font.Font(None, 28)
        num_remaining_tasks = len([task for task in self.tasks if task[self.TASK_STATUS] == self.AVAILABLE])
        text = font.render(f"Remaining Tasks: {num_remaining_tasks}", True, self.WHITE)
        self.screen.blit(text, (stat_left_column_X, self.LOCATIONS_RIGHT_START_Y + 320))

        num_free_devices = len([device for device in self.devices if device[self.DEVICE_STATUS] == self.AVAILABLE])
        text = font.render(f"Free Devices: {num_free_devices}", True, self.WHITE)
        self.screen.blit(text, (stat_left_column_X, self.LOCATIONS_RIGHT_START_Y + 360))

        num_free_agents = self.agents.query(f"status == {self.AGENT_AVAILABLE}").shape[0]
        text = font.render(f"Free Agents: {num_free_agents}", True, self.WHITE)
        self.screen.blit(text, (stat_right_column_X, self.LOCATIONS_RIGHT_START_Y + 360))

        # Draw a horizontal line at the end of dynamic stats with a width of stat box
        pygame.draw.line(self.screen, self.WHITE, (self.R_STAT_BOX_X, self.LOCATIONS_RIGHT_START_Y + 400), \
                         (self.R_STAT_BOX_X + self.R_STAT_BOX_WIDTH, self.LOCATIONS_RIGHT_START_Y + 400), 2)
        
        # show the time step after about line and at the center of stat box
        font = pygame.font.Font(None, 28)
        text = font.render(f"Time Step", True, self.WHITE)
        self.screen.blit(text, (stat_left_column_X, self.LOCATIONS_RIGHT_START_Y + 420))

        font = pygame.font.Font(None, 38)
        text = font.render(f"{self.time_step}", True, self.WHITE)
        self.screen.blit(text, (stat_left_column_X + self.R_STAT_BOX_WIDTH / 4, self.LOCATIONS_RIGHT_START_Y + 418))


   
    def _render_pygame(self):
        self.screen.fill(self.WHITE) # fill the screen with white color
        self._render_locations() # locations means shelves (these are static - i.e does not move)
        self._render_devices()
        self._render_agents()
        self._render_stats()

        pygame.display.flip()
        pygame.time.delay(self.STEP_DURATION) # wait time

    
    def render(self, mode='human'):
        if mode == 'human':
            self._render_pygame()
        else:
            print(f"=======> [from env - render]: {mode} not supported at this moment")
            
    
    def close(self):
        pass

    def get_observation(self):
        return {
            "tasks": self.tasks,
            "devices": self.devices
        }
    
