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
# import pymc3 as pm
# import theano.tensor as tt
# import arviz as az

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