import numpy as np
import pandas as np

class dataset:

    data = []

    def __init__(self):
        print("dataset initialised !!!")

    def __new__(self):
        print("dataset created !!!")

    def get_csv(csv_file : str = None):
        if csv_file != None:
            data = pd.read_csv(csv_file)
        else :
            print("please provide a valid path to a csv file")

