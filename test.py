# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:22:50 2022

@author: User
"""

import pandas as pd
import numpy as np

data = pd.read_excel('crops_info.xlsx', index_col=0)

modDfObj = data.apply(lambda x: np.random.normal(x,0.1*x) if x.name == 'Revenue (R$/ton)' else x, axis=1)
modDfObj = modDfObj.apply(lambda x: np.random.normal(x,0.1*x) if x.name == 'Cost (R$/ton)' else x, axis=1)
weights = modDfObj.loc['Revenue (R$/ton)'] - modDfObj.loc['Cost (R$/ton)']
probabilities = weights / weights.sum()