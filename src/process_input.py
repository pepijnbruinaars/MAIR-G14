# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:13:31 2023

@author: 13vic
"""
import pandas as pd
import sys
import re
sys.path.insert(0, 'C:/Users/13vic/Desktop/MAIR/MAIR-G14/src/mlp_model')

from random_forest import process_data

information = pd.read_csv("../data/restaurant_info.csv")



area = ['west', 'north', 'south', 'centre', 'east']
food = ['moderate', 'expensive', 'cheap']
price = ['british', 'modern european', 'italian', 'romanian', 'seafood',
       'chinese', 'steakhouse', 'asian oriental', 'french', 'portuguese',
       'indian', 'spanish', 'european', 'vietnamese', 'korean', 'thai',
       'moroccan', 'swiss', 'fusion', 'gastropub', 'tuscan',
       'international', 'traditional', 'mediterranean', 'polynesian',
       'african', 'turkish', 'bistro', 'north american', 'australasian',
       'persian', 'jamaican', 'lebanese', 'cuban', 'japanese', 'catalan']



def extract_prefrence(input_string : str):
    
    i
    print("hello world")
    