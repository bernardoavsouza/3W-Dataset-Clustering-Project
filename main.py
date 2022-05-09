# -*- coding: utf-8 -*-


import functions as f
import numpy as np
import pandas as pd


SETTINGS = f.load_settings()


def main():
   data, labels = f.load_data(SETTINGS)
   model = f.main_processing(SETTINGS, data, labels)
   accuracy = f.evaluation(SETTINGS, model, data, labels) 
   
   return accuracy    
    

    
if __name__ == '__main__':
    accuracy = main()
