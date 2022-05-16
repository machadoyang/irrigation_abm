# -*- coding: utf-8 -*-
"""
@author: machadoyang
"""

from mesa import Agent, Model
from mesa.time import StagedActivation
import networkx as nx
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from scheduler import AgentTypeScheduler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import random

import data_preparation

class FarmerAgent(Agent):
    """An agent that chooses a crop, asks for water and interact with water canal."""
    _last_id = 0
    def __init__(self, model):
        super().__init__(FarmerAgent._last_id+1, model)
        FarmerAgent._last_id += 1
        self.type = 'farmer'
        self.poverr = 0.3  # Probability of override
        self.pcrop = [0.55, 0.4, 0.05]  # Proabability of crop choice
        self.cropChoice = None
        self.area = 50 # in hectares
        self.water_rights_quantity = 0
        self.water_rights_lognorm_fit = data_preparation.fit_water_rights()
        self.life_time = 0 # water right life time in years
        
    def cropsselection(self):
        list_of_crops = ['Rice', 'Maize', 'Soya']
        self.cropChoice = np.random.choice(list_of_crops, 1, p = self.pcrop)
        
    def water_right_random(self):
        ss.lognorm.rvs(loc=self.water_rights_lognorm_fit['loc'], scale=self.water_rights_lognorm_fit['scale'])
        
    def check_neighbours(self):
        """ Check neighbours content """
        neighbors_nodes = self.model.grid.get_neighbors(self.pos)
        test_neighbours = [agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)]
        # print(neighbors_nodes[1])

        # if len(neighbors_nodes) > 1:
        #     other = self.random.choice(neighbors_nodes)
        #     return(neighbors_nodes)
            # other.testvar += 1
            # self.testvar -=1
        # for agent in self.model.grid.get_cell_list_contents(neighbors_nodes):
        #     continue
                    
        # list_of_contents = self.model.grid.get_cell_list_contents(neighbors_nodes).cropChoice
        # print(list_of_contents)
    
    def ask_water(self):
        # self.amount_water = self.area*10 # value in m3/s
        self.amount_water = ss.lognorm.rvs(s=self.water_rights_lognorm_fit['shape'], loc=self.water_rights_lognorm_fit['loc'], scale=self.water_rights_lognorm_fit['scale'])*12 # Estimate monthly amount of water and multiply by 12 to get yearly value
        print("Farmer asked water")
        print(self.amount_water)
        
    def demandwater(self):
        pass
    
    def inactivate(self, model, prob=0.05):
        """ Remove an agent given a probability each year

        Args:
            prob: Probability of agent be killed at each step.
            model: Model object.

        """
        if np.random.binomial(1,prob) != 0: # 10% chance of dying
            model.schedule.remove(self)

    def step(self):
        if (self.life_time == 0):
            self.cropsselection()
        self.inactivate(model, 0.05)
        self.ask_water()
        self.life_time += 1 # Increase 1 year in water right lite time
        print ("Hi, I am farmer n. " + str(self.unique_id) + " and chose to plant " + str(self.cropChoice[0]) + "." + " It has passed " + str(self.life_time) + " years.")
        # print(self.model.annual_flow_discharge)
        # self.check_neighbours()
        # a = model.grid.get_neighbors(self.pos, include_center=False)
        # print ("My neighboor cells are: " + self.neighbors_nodes)
        
class ManagerAgent(Agent):
    """An agent that allocate water to farmers."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'manager'

    # Allocates water based on the agent id (upstream comes first)
    def allocate_water_fcfs(self):
        print("Manager is concieving water rights on the basis of first come first served.")
        farmers_contents = self.model.grid.get_all_cell_contents()
        print("---")
        print(farmers_contents)
        for farmer in farmers_contents:
            if farmer.type == 'farmer':
                farmer_section = model.G.nodes[farmer.pos]["section"] # get section information where farmer is positioned
                print(model.available_water_per_section[str(farmer_section)])
                if model.available_water_per_section[str(farmer_section)] >= farmer.amount_water:
                    # Concieve water right and deduct from available water per section
                    farmer.water_rights_quantity = farmer.amount_water
                    model.available_water_per_section[str(farmer_section)] -= farmer.amount_water
                    print("Farmer n. " + str(farmer.unique_id) + " received " + str(farmer.water_rights_quantity) + " m³/year")
                    print ("Remaining water available on section " + str(farmer_section) + " is " + str(model.available_water_per_section[str(farmer_section)]))
                else:
                    print ("Water request denied to farmer " + str(farmer.unique_id))
                    print("There is no available water at section " + str(farmer_section))
        #             # Concieve water rights
        #             cell.water_rights_quantity = cell.amount_water
        #             print("Farmer n. " + str(cell.unique_id) + " received " + str(cell.water_rights_quantity) + " m³/year.")
        #             # Deduct water quantity from total
        #             model.available_water_to_distribute -= cell.amount_water
        #         elif model.available_water_to_distribute == 0:
        #             cell.water_rights_quantity = 0
        #             print("Querido farmer n. " + str(cell.unique_id) + ". Vai ficar sem esse ano, lindeza. Bjs de luz")
        #         else:
        #             cell.water_rights_quantity = model.available_water_to_distribute
        #             print("Farmer n. " + str(cell.unique_id) + " recieved " + str(cell.water_rights_quantity) + " m³/year.")
        #             model.available_water_to_distribute = 0
                
    def step(self):
        self.allocate_water_fcfs()


class IrrigationModel(Model):
    """
    The Irrigation Model for Water Allocation 
    """
    
    verbose = True  # Print-monitoring
    current_id = 0
    
    def __init__(
            self,
            linear_graph,
            hydro_stats,
            crop_characteristics,
            water_rights_gamma_fit,
            available_water_per_section):
        # self.schedule = StagedActivation(self)
        self.schedule = AgentTypeScheduler(IrrigationModel, [FarmerAgent, ManagerAgent])
        
        """
        Create a new Irrigation model with the given parameters.
        
        Args:
            linear_graph: generated linear graph using networkx
            hydro_stats: statistics to generate annual hydrological data
            crop_characteristics: contain cost, return, max yield and max profit for each crop
        """ 
    
        # Set parameters
        self.G = linear_graph
        self.grid = NetworkGrid(self.G)
        self.hydro_stats = hydro_stats
        self.water_rights_gamma_fit = water_rights_gamma_fit
        self.available_water_per_section = available_water_per_section
        
        # Create farmer agents and position them in order
        def create_farmers_in_order(self):
            for i,node in enumerate(self.G.nodes()):
                f = FarmerAgent(i+1, self)
                self.schedule.add(f)
                self.grid.place_agent(f, node)
                          
        # Create manager agent
        def create_manager(self):
            m = ManagerAgent(len(self.G.nodes())+1, self)
            self.schedule.add(m)
        
        create_manager(self)
        # self.datacollector = DataCollector(
        # agent_reporters={"testvar": lambda f: f.cropChoice(FarmerAgent)})
        # Create farmer agents and position them randomly
    def create_farmers_random_position(self):
        """
        Create farmers at random position based on normal distribution.
        """ 
        # self.number_of_farmers_to_create = len(linear_graph.nodes())
        self.number_of_farmers_to_create = round(0.05*len(linear_graph.nodes()))
        self.number_of_farmers_to_create = -1
        while (self.number_of_farmers_to_create < 0):
            # self.number_of_farmers_to_create = round(np.random.normal(loc=10, scale=2))
            self.number_of_farmers_to_create = 2
        
        i = 0
        while (i < self.number_of_farmers_to_create):
            random_node = random.sample(list(linear_graph.nodes()), 1)
            if (len(self.grid.get_cell_list_contents(random_node)) == 0): # Check whether cell is empty. If so, place agent
                f = FarmerAgent(self)
                self.schedule.add(f)
                self.grid.place_agent(f, random_node[0])
                i+=1
                
        # print(self.grid.get_cell_list_contents([i+1]))
        # model.grid.get_all_cell_contents()
        # random_nodes = random.sample(list(linear_graph.nodes()), self.number_of_farmers_to_create)
        # print("Water rights order for each farmer in this run: " + str(random_node))
        # for i,node in enumerate(random_nodes):
        #     f = FarmerAgent(i+1, self)
        #     self.schedule.add(f)
        #     self.grid.place_agent(f, node)
        
    # def generate_annual_hydro_data(self):
    #     z1 = np.random.normal() # independent random variable with the standard normal distribution (flow discharge)
    #     z2 = np.random.normal() # independent random variable with the standard normal distribution (rainfall)
    #     self.annual_flow_discharge = self.hydro_stats.mu_x + self.hydro_stats.sigma_x*z1
    #     self.annual_rainfall = self.hydro_stats.mu_y + self.hydro_stats.sigma_y*(z1*self.hydro_stats.rho + z2*np.sqrt(1-self.hydro_stats.rho**2))
    #     self.available_water_to_distribute = self.annual_flow_discharge
    #     print("In this year, annual flow discharge is " + str(self.annual_flow_discharge) + " m³/year and annual total rainfall is " + str(self.annual_rainfall) + " mm/year.")
    #     # print(self.annual_flow_discharge)
    #     # print(self.annual_rainfall)
        
        # annual_flow_discharge = self.mu_x + sigma_x*
        
    def generate_annual_water_availability_canal(self):
        self.annual_flow_discharge = 252000*365 # m³/day multiplied by number of days
        self.percentage_to_distribute = 0.2
        self.available_water_to_distribute = self.percentage_to_distribute*self.annual_flow_discharge #percentage of total available to distribute
        # print("In this year, available annual flow discharge is " + str(self.available_water_to_distribute))
            
    def step(self):
        """ Execute the step of all the agents, one at a time. At the end advance model by one step """
        # self.datacollector.collect(self)
        self.create_farmers_random_position()
        self.generate_annual_water_availability_canal()
        # self.generate_annual_hydro_data()
        self.schedule.step()
        
    def run_model(self, step_count=2):
        
        # if self.verbose:
            # print("Number of farmers: ", self.schedule.get_breed_count(Wolf))
            
        for i in range(step_count):
            print ("-------------- \n" +
                   "Initiating year n. " + str(i) + "\n" +
                   "--------------")
            self.step()
        
"Generate Linear Graph with NX"

def get_evenly_divided_values(value_to_be_distributed, times):
    """
    Divide a number into (almost) equal whole numbers
    
    Args:
        value_to_be_distributed: number to be divided
        times: Number of integer values to divide
    """ 
    return [value_to_be_distributed // times + int(x < value_to_be_distributed % times) for x in range(times)]

def generate_edges_linear_graph(number_of_sections = 10, number_of_nodes=25):
    """
    Create a linear graph with 'section' attribute (almost) evenly spaced
    according to number of desired number of edges
    
    Args:
        number_of_sections: Number of sections to devide nodes
        number_of_nodes: Number of nodes (farmers possible positions)
    """ 
    
    n_nodes_per_section = get_evenly_divided_values(number_of_nodes, number_of_sections)
    sections_list = []
    for i, v in enumerate(n_nodes_per_section):
        for j in range(v):
            sections_list.append(i+1)   
    sections = {i+1: {'section': v} for i, v in enumerate(sections_list)}
    edges = []
    for x in range(1, number_of_nodes):
        edges.append((x,x+1))
    linear_graph = nx.Graph()
    linear_graph.add_edges_from(edges)
    nx.set_node_attributes(linear_graph, sections)
    nx.draw_networkx(linear_graph)
    return linear_graph

linear_graph = generate_edges_linear_graph(number_of_sections = 10, number_of_nodes=10)

"""Sections water availability information"""

available_water_per_section = {
    '1': 100001,
    '2': 100002,
    '3': 100003,
    '4': 100004,
    '5': 100005,
    '6': 100006,
    '7': 100007,
    '8': 100008,
    '9': 100009,
    '10': 100010,
    }

"Initial conditions"
Pc = [0.55, 0.4, 0.05] # Crop choice: [rice, maize, soya]
Povr = 0.3 # Prob of override

# "Stats properties of the hydrological input conditions"

hydro_stats = pd.Series({'mu_x': 3191, 'sigma_x': 725.95, 'mu_y': 130, 'sigma_y': 30.82, 'rho': 0.98})
crop_characteristics = pd.DataFrame.from_dict({'rice': [2, 0.55, 30, 16.5], 'maize': [1, 0.5, 20, 10], 'soya': [0.8, 1.25, 6, 7.5]}, columns=['cost', 'return', 'maxYield', 'maxProfit'], orient='index')

"Run model"
water_rights_gamma_fit = data_preparation.read_water_rights()
model = IrrigationModel(linear_graph, hydro_stats, crop_characteristics, water_rights_gamma_fit, available_water_per_section)
model.run_model()
# test = model.datacollector.get_agent_vars_dataframe()
# steps = 11
# for i in range(steps):
#     model.step()
# agent_state = model.datacollector.get_model_vars_dataframe()