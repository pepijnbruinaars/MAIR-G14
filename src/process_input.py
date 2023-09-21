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
price = ['modern european', 'italian', 'romanian', 'seafood',
       'chinese', 'steakhouse', 'asian oriental', 'french', 'portuguese',
       'indian', 'spanish', 'european', 'vietnamese', 'korean', 'thai',
       'moroccan', 'swiss', 'fusion', 'gastropub', 'tuscan',
       'international', 'traditional', 'mediterranean', 'polynesian',
       'african', 'turkish', 'bistro', 'north american', 'australasian',
       'persian', 'jamaican', 'lebanese', 'cuban', 'japanese', 'catalan']



def extract_preference(input_string : str):
    
    # make sure input is in lower case
    input_string = input_string.lower()
    
    # First entry
    food_regex = "british"
    
    # for every other entry add the option of food
    for i in price:
        food_regex = food_regex + "|" + i 
    
    # match the possible preferences to the input
    area_match = re.search(r"west|north|south|centre|east", input_string)
    food_match = re.search(r"moderate|expensive|cheap", input_string)
    price_match = re.search(rf"{food_regex}", input_string)

    
    if food_match:
        print(food_match.group())
        
    if area_match:
        print(area_match.group())
    
    if price_match:
        print(price_match.group())
        
    return area_match.group()
    
    
    
    
    
if __name__ == "__main__":
    extract_preference("I want a restaurant that italian is in the middle or centre of the city")