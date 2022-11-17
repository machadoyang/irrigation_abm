# -*- coding: utf-8 -*-
"""
@author: machadoyang
"""

from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from scheduler import AgentTypeScheduler

import numpy as np
import pandas as pd
import scipy.stats as ss
import random

from collections import defaultdict

import data_preparation


class FarmerAgent(Agent):
    """An agent that chooses a crop, asks for water and interact with water canal."""
    _last_id = 0

    def __init__(self, model):
        super().__init__(FarmerAgent._last_id+1, model)
        FarmerAgent._last_id += 1
        self.type = 'farmer'
        self.pcrop = [0.6, 0.2, 0.2]  # Proabability of crop choice
        self.harvest_efficiency = 0  # Amount of harvest based on chosen crop in ton/ha
        self.crop_yield_per_ton = 0  # Chosen crop yield per ton in R$/ton
        self.cropChoice = None
        self.area = 0
        self.amount_of_water_asked = 0
        self.amount_of_water_received = 0
        self.p_to_override = np.random.beta(a=2, b=5, size=1)[
            0]  # Probability to override
        self.amount_of_water_withdrawn = 0
        self.yearly_expected_yield = 0
        self.total_revenue = 0
        self.water_rights_lognorm_fit = model.water_rights_parameters
        self.life_time = 0  # water right life time in years
        self.received_water_right = False
        self.agent_performed_override = False
        self.agent_water_scarcity = False

    def crops_selection(self):
        # Calculate weights of choosing a crop based on expected profit
        weights = model.crops_info_year.loc['Revenue (R$/ton)'] - \
            model.crops_info_year.loc['Cost (R$/ton)']
        # Calculates probabilities of choosing a crop
        probabilities = weights / weights.sum()
        self.cropChoice = np.random.choice(
            crops_info.columns.values, 1, p=probabilities)[0]  # Choose crop
        self.harvest_efficiency = model.crops_info_year.loc['Yield (ton/ha)'][self.cropChoice]
        self.crop_yield_per_ton = model.crops_info_year.loc[
            'Revenue (R$/ton)'][self.cropChoice]

    def expected_annual_yield(self):
        self.yearly_expected_yield = self.area * \
            self.harvest_efficiency * self.crop_yield_per_ton

    def water_right_random(self):
        ss.lognorm.rvs(
            loc=self.water_rights_lognorm_fit['loc'], scale=self.water_rights_lognorm_fit['scale'])

    def check_neighbours(self):
        """ Check neighbours content """
        neighbors_nodes = self.model.grid.get_neighbors(self.pos)
        test_neighbours = [
            agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)]

    def define_initial_area(self):
        monthly_amount_of_water_based_on_data = ss.lognorm.rvs(
            s=self.water_rights_lognorm_fit['shape'], loc=self.water_rights_lognorm_fit['loc'], scale=self.water_rights_lognorm_fit['scale'])
        self.area = monthly_amount_of_water_based_on_data/(40*30)

    def calculate_amount_of_water_to_ask(self):
        # Area times 40 m³/ha/day times 30 days and 12 months to get yearly value
        self.amount_of_water_asked = self.area*40*30*12


    def inactivate(self, model, prob=0.05):
        """ Remove an agent given a probability each year

        Args:
            prob: Probability of agent be killed at each step.
            model: Model object.

        """
        if np.random.binomial(1, prob) != 0:  # 10% chance of dying
            model.schedule.remove(self)

    def step(self):
        if (self.life_time == 0):
            self.define_initial_area()
        self.crops_selection()
        # self.inactivate(model, 0.05)
        self.calculate_amount_of_water_to_ask()

        self.expected_annual_yield()
        print("Hi, I am farmer n. {} and chose to plant {}. I have asked {:.2f} m³/year. It has passed {} years since I created my farm.".format(
            self.unique_id, self.cropChoice, self.amount_of_water_asked, self.life_time))

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
        if model.verbose == True:
            print(
                "Manager is conceiving water rights on the basis of first come first served.")
            print("Distributing water")
        agents_contents = self.model.grid.get_all_cell_contents()
        # sort agents by unique_id
        agents_contents.sort(key=lambda x: x.unique_id)
        for agent in agents_contents:
            if (agent.type == 'farmer' and agent.life_time == 0):
                # get section information where farmer is positioned
                farmer_section = model.G.nodes[agent.pos]["section"]
                if (model.virtual_water_available_per_section[str(farmer_section)] >= agent.amount_of_water_asked):
                    # Conceive water right and deduct from available water per section
                    agent.amount_of_water_received = agent.amount_of_water_asked
                    model.virtual_water_available_per_section[str(
                        farmer_section)] -= agent.amount_of_water_asked
                    agent.received_water_right = True
                    if model.verbose == True:
                        print("Farmer n. " + str(agent.unique_id) + " received " +
                              str(agent.amount_of_water_received) + " m³/year")
                        print("Remaining virtual water available on section " + str(farmer_section) +
                              " is " + str(model.virtual_water_available_per_section[str(farmer_section)]))
                else:
                    if model.verbose == True:
                        print("Water request denied to farmer {}. There is no available water at section {}.". format(
                            agent.unique_id, farmer_section))

    def water_withdrawal(self):
        """
        Withdraws water from the canal upstream to downstream.
        To calculate water balance transfer all water to the first section.
        In a for loop move the remaining water to the next section
        """
        agents_contents = self.model.grid.get_all_cell_contents()
    
        # Group agents by section position
        grouped_agents = defaultdict(list)
        for agent in agents_contents:
            grouped_agents[model.G.nodes[agent.pos]["section"]].append(agent)
        
        # Transfer all water from first section to first element in grouped_agents.keys()
        # Algorithm will break otherwise in case no agent allocates itself in section 1 on the first iteration
        current_list_of_sections = list(grouped_agents.keys())
        model.available_water_per_section[str(current_list_of_sections[0])] = model.available_water_per_section[str(1)]
        
        for section in grouped_agents:
            for agent in grouped_agents[section]:
                if (agent.type == 'farmer'):
                    # get section information where farmer is positioned
                    farmer_section = model.G.nodes[agent.pos]["section"]
                    if (model.available_water_per_section[str(farmer_section)] >= agent.amount_of_water_asked):
                        if (agent.received_water_right == True):
                            agent.amount_of_water_withdrawn = agent.amount_of_water_received
                            model.available_water_per_section[str(
                                farmer_section)] -= agent.amount_of_water_received
                        else:
                            if (agent.p_to_override < model.override_threshold):
                                agent.agent_performed_override = True
                                agent.amount_of_water_withdrawn = agent.amount_of_water_asked
                                model.available_water_per_section[str(
                                    farmer_section)] -= agent.amount_of_water_received
                                if model.verbose == True:
                                    print("Agent {} have overriden manager's decision withdrawing {} m³/year.". format(
                                        agent.unique_id, agent.amount_of_water_withdrawn))
                    else:
                        agent.agent_water_scarcity = True
                        agent.amount_of_water_withdrawn = 0
                    # Calculate agent total yield based on water received
                    agent.total_revenue = 10 * agent.amount_of_water_withdrawn / \
                        (40*30*12) * agent.harvest_efficiency * \
                        agent.crop_yield_per_ton
                agent.life_time += 1  # Increase 1 year in water right life time
            if (section != list(grouped_agents.keys())[-1] ):
                next_section = current_list_of_sections[current_list_of_sections.index(section)-len(current_list_of_sections)+1]
                model.available_water_per_section[str(next_section)] += model.available_water_per_section[str(section)]

    def step(self):
        self.allocate_water_fcfs()
        self.water_withdrawal()


class IrrigationModel(Model):
    """
    The Irrigation Model for Water Allocation 
    """

    verbose = True  # Print-monitoring
    current_id = 0

    def __init__(
            self,
            linear_graph,
            water_rights_parameters,
            init_water_available_per_section,
            crops_info,
            n_farmers_to_create_per_year,
            override_threshold):
        # self.schedule = StagedActivation(self)
        self.schedule = AgentTypeScheduler(
            IrrigationModel, [FarmerAgent, ManagerAgent])

        """
        Create a new Irrigation model with the given parameters.
        
        Args:
            linear_graph: generated linear graph using networkx
            water_rights_parameters: water rights dataset to fit a lognormal distribution
            available_water_per_section: dictionary containing amount of water available at each section
            crops_info: pandas DataFrame containing crops information (revenue, yield and cost)
            n_farmers_to_create_per_year: int number that represent the number of farmers to create per year
        """

        # Set parameters
        self.G = linear_graph
        self.grid = NetworkGrid(self.G)
        self.water_rights_parameters = water_rights_parameters
        # what has been conceived by water agency
        self.init_water_available_per_section = init_water_available_per_section
        self.virtual_water_available_per_section = init_water_available_per_section
        # what is actually available in canal based on water withdrawal (counting water right overrides)
        self.available_water_per_section = available_water_per_section
        self.init_water_available_per_section = self.allocate_all_water_to_section_one().copy()
        self.override_threshold = override_threshold

        self.df_model_variables = pd.DataFrame(
            columns=['Step', 'Section', 'Water available', 'Virtual Water Available'])

        # Create farmer agents and position them in order
        def create_farmers_in_order(self):
            for i, node in enumerate(self.G.nodes()):
                f = FarmerAgent(i+1, self)
                self.schedule.add(f)
                self.grid.place_agent(f, node)

        # Create manager agent
        def create_manager(self):
            m = ManagerAgent(len(self.G.nodes())+1, self)
            self.schedule.add(m)

        create_manager(self)

        # Data Collector
        self.datacollector = DataCollector(
            agent_reporters={
                "Type":
                    lambda x: x.type,
                "Position":
                    lambda x: x.pos if x.type == 'farmer' else None,
                "Planned crop area (ha)":
                    lambda x: x.area if x.type == 'farmer' else None,
                "Crop choice":
                    lambda x: x.cropChoice if x.type == 'farmer' else None,
                "Amount of water asked (m³/year)":
                    lambda x: x.amount_of_water_asked if x.type == 'farmer' else None,
                "Amount of water received (m³/year)":
                    lambda x: x.amount_of_water_received if x.type == 'farmer' else None,
                "Received water right":
                    lambda x: x.received_water_right if x.type == 'farmer' else None,
                "Total revenue (R$)":
                    lambda x: x.total_revenue if x.type == 'farmer' else None,
                "Amount of water withdrawn (m³/year)":
                    lambda x: x.amount_of_water_withdrawn if x.type == 'farmer' else None,
                "Crop Choice":
                    lambda x: x.cropChoice if x.type == 'farmer' else None,
            },
        )
        self.datacollector.collect(self)

    def allocate_all_water_to_section_one(self):
        init_water = self.init_water_available_per_section.copy()
        # sum total water available in m³/year
        total_water_available = sum(init_water.values())
        actual_water_available = {}
        for i in init_water:
            if (i == '1'):
                actual_water_available[i] = total_water_available
            else:
                actual_water_available[i] = 0
        return actual_water_available

        # Create farmer agents and position them randomly
    def create_farmers_random_position(self):
        """
        Create farmers at random position in a linear graph.
        """
        self.number_of_farmers_to_create = round(
            0.05*len(linear_graph.nodes()))
        self.number_of_farmers_to_create = n_farmers_to_create_per_year

        i = 0
        while (i < self.number_of_farmers_to_create):
            random_node = random.sample(list(linear_graph.nodes()), 1)
            # Check whether cell is empty. If so, place agent
            if (len(self.grid.get_cell_list_contents(random_node)) == 0):
                f = FarmerAgent(self)
                self.schedule.add(f)
                self.grid.place_agent(f, random_node[0])
                i += 1

        # print(self.grid.get_cell_list_contents([i+1]))
        # model.grid.get_all_cell_contents()
        # random_nodes = random.sample(list(linear_graph.nodes()), self.number_of_farmers_to_create)
        # print("Water rights order for each farmer in this run: " + str(random_node))
        # for i,node in enumerate(random_nodes):
        #     f = FarmerAgent(i+1, self)
        #     self.schedule.add(f)
        #     self.grid.place_agent(f, node)

    def fluctuate_market_for_the_year(self):
        def calculate():
            self.crops_info_year = crops_info.apply(lambda x: np.random.normal(
                x, 0.05*x) if x.name == 'Revenue (R$/ton)' else x, axis=1)  # apply random variation on revenue
            self.crops_info_year = self.crops_info_year.apply(lambda x: np.random.normal(
                x, 0.05*x) if x.name == 'Cost (R$/ton)' else x, axis=1)  # apply random variation on cost
        calculate()
        profit_info_year = self.crops_info_year.loc['Revenue (R$/ton)'] - \
            self.crops_info_year.loc['Cost (R$/ton)']
        while (any(profit_info_year < 0)):
            calculate()

    def reset_water_available_for_current_year(self):
        self.available_water_per_section = self.allocate_all_water_to_section_one().copy()

    def collect_model_attributes(self):
        for key in self.virtual_water_available_per_section:
            self.df_model_variables = self.df_model_variables.append(
                {
                    'Step': model.schedule.steps,
                    'Section': key,
                    'Water available': self.available_water_per_section[key],
                    'Virtual Water Available': self.virtual_water_available_per_section[key],
                },
                ignore_index=True
            )

    def step(self):
        """ Execute the step of all the agents, one at a time. At the end advance model by one step """
        # Preparation
        self.create_farmers_random_position()
        self.fluctuate_market_for_the_year()
        if (model.schedule.steps == 0):
            self.reset_water_available_for_current_year()
        # Run step
        self.schedule.step()

        # Save model variables
        self.datacollector.collect(self)
        self.collect_model_attributes()
        
        # After step
        self.reset_water_available_for_current_year()

    def run_model(self, step_count=3):
        for i in range(step_count):
            print("-------------- \n" +
                  "Initiating year n. " + str(i) + "\n" +
                  "--------------")
            self.step()


"Generate Linear Graph with NX"
linear_graph = data_preparation.generate_edges_linear_graph(
    number_of_sections=15, number_of_nodes=10000)

"Initial conditions"
# Values in m³/year
available_water_per_section = {
    '1': 764.5*4320,
    '2': 808.1*4320,
    '3': 752.4*4320,
    '4': 825.1*4320,
    '5': 784.2*4320,
    '6': 680.0*4320,
    '7': 646.7*4320,
    '8': 569.9*4320,
    '9': 518.3*4320,
    '10': 435.9*4320,
    '11': 377.8*4320,
    '12': 344.2*4320,
    '13': 265.9*4320,
    '14': 261.8*4320,
    '15': 305.0*4320,
}

# Read crops information
crops_info = pd.read_excel('crops_info.xlsx', index_col=0)
#Number of farmers to create per year
n_farmers_to_create_per_year = 101
# number of steps to run
number_of_steps = 20
# Get parameters estimation to fit water rights requests
water_rights_parameters = data_preparation.fit_water_rights()

""""Override Threshold"""
override_threshold = 0.1

"Run model"
model = IrrigationModel(linear_graph,
                        water_rights_parameters,
                        available_water_per_section,
                        crops_info,
                        n_farmers_to_create_per_year,
                        override_threshold)
model.run_model(step_count=number_of_steps)

"Collect model variables"
agents_results = model.datacollector.get_agent_vars_dataframe()
model_results = model.df_model_variables
