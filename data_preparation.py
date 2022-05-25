# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:17:36 2022

@author: User
"""

import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import scipy.stats as ss
import networkx as nx

def read_water_rights():
    """Read .xslx containing water rights from Canal do Sertão and group into monthly discharge"""
    
    data = pd.read_excel('C:/Users/User/OneDrive/Doutorado/Canal do Sertão/CANAL DO SERTÃO_SEMARH.xlsx', sheet_name='CADASTRO DE USUARIOS', skiprows=5)
    water_rights = data[~data['SITUAÇÃO'].isin(['ANÁLISE', 'ANÁLISE PROCESSO SEI'])]
    water_rights_mothly_grouped = water_rights['VOLUME DIÁRIO (m³)']*water_rights['PERÍODO (dias/mês)']
    water_rights_mothly_grouped = water_rights_mothly_grouped.dropna()
    return water_rights_mothly_grouped
   
def fit_water_rights():
    """Fit a lognormal distribution using Maximum Likelihood Estimators"""
    water_rights = read_water_rights()
    shape, loc, scale = ss.lognorm.fit(water_rights)
    return {'shape': shape, 'loc': loc, 'scale': scale}

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
    according to number of edges
    
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
