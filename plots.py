# -*- coding: utf-8 -*-
"""
@author: machadoyang
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np

plt.style.use('seaborn-white')

def agents_impact_over_years(agents_results):
    farmers_results = agents_results[agents_results['Type'] == 'farmer']
    fig, ax = plt.subplots(dpi=300)
    for i in range(farmers_results.index.get_level_values(1).min(), farmers_results.index.get_level_values(1).max()):
        results_current_agent = farmers_results.xs(i, level=1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('step')
        ax.set_ylabel('Total profit (R$)')
        ax.set_yscale('log')
        ax.plot(results_current_agent['Total profit (R$)'].index,
                 results_current_agent['Total profit (R$)'].values, color='gray', alpha=0.3)
        print(results_current_agent)
        
def water_level_in_canal(model_results):
    fig, ax = plt.subplots(dpi=300)
    for i in range (1, 11): # number of sections
        current_array = []
        model_results['available_water_array'].apply(lambda x: current_array.append(x[str(i)]))
        print(current_array)
        print(np.repeat(i, len(current_array)))
        ax.plot(current_array, np.repeat(i, len(current_array)))
        # print(i)
        # print(len(results_current_agent))
        # if len(results_current_agent) > 1:
        #     plt.plot(results_current_agent['Total profit (R$)'].index,
        #              results_current_agent['Total profit (R$)'].values)
    # step_vs_agents = pd.DataFrame(
    #     farmers_results.index.get_level_values(0).min(), index=np.arange(farmers_results.index.get_level_values(0).max()),
    #     columns=np.arange(1, farmers_results.index.get_level_values(1).max() + 1))
    # # loop over steps count
    # for i in range(farmers_results.index.get_level_values(0).min(), farmers_results.index.get_level_values(0).max() + 1):
    #     current_step_results = farmers_results.loc[i]
    #     # columns = current_step_results.columns
    #     for j in step_vs_agents.columns:
    #         print(i, j)
    #         # try:
    #         #     print(current_step_results.loc[j])
    #         # except: continue
    #         # step_vs_agents.at['0','0'] = current_step_results.loc[j]
    #         try:
    #             step_vs_agents.at[i,j] = current_step_results.loc[j]
    #         except:
    #             step_vs_agents.at[i,j] = 0