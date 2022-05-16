# -*- coding: utf-8 -*-
"""
@author: machadoyang
"""


from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid # SingleGrid enforces at most one agent per cell while MultiGrid allows multiple agents to be in the same cell
from mesa.datacollection import DataCollector

import matplotlib.pyplot as plt
import numpy as np

# The model class holds the model-level attributes, manages the agents,
# and generally handles the global level of our model

def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return (1 + (1/N) - 2*B)

class MoneyAgent(Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model) # Good practice to give each agent an unique id
        self.wealth = 1
        
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False) # moore = True inclui diagonais
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        
    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos]) # get_cell_list_contents get contents of one or more cells by passing tuples
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1
        
    def step(self):
        # self.move()
        if self.wealth > 0:
            self.give_money()
        
        
        
class MoneyModel(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)  # Simple 
        
        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y)) # place agents on the grid
            
    # The scheduler is a component which controls the order in which agents are activated
    # For example, all the agents may activate in the same order every step; their order
    # might be shuffled; we may try to simulate all the agents acting at the same time, etc...
            
        self.datacollector = DataCollector(
            model_reporters={"Gini": compute_gini},
            agent_reporters={"Wealth": "wealth"})
      
    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self) # at every step we will collect data and calculate Gini coeff.
        self.schedule.step()
        
"""Multiple runs to see how model behaves"""
all_wealth = []
for j in range(100):
    model = MoneyModel(50, 10, 10)
    for i in range(20):
        model.step()
    
    # Store the results
#    for agent in model.schedule.agents:
#        all_wealth.append(agent.wealth)
    
agent_counts = np.zeros((model.grid.width, model.grid.height))
for cell in model.grid.coord_iter(): # loop over every cell in the grid, giving us each cell’s coordinates and contents in turn
    cell_content, x, y = cell
    agent_count = len(cell_content)
    agent_counts[x][y] = agent_count
plt.imshow(agent_counts, interpolation='nearest')
plt.colorbar()

gini = model.datacollector.get_model_vars_dataframe()
gini.plot()

agent_wealth = model.datacollector.get_agent_vars_dataframe()
agent_wealth.head()