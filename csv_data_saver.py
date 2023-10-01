import csv
import pandas as pd
import math
import numpy as np

time = pd.DataFrame()
accel = pd.DataFrame()
velo = pd.DataFrame()
alt = pd.DataFrame()

data = pd.read_csv("data/flight_data_2 - flight_data_2.csv.csv")
time = data['time']
accel = data['accel']
velo = data['velo']
alt = data['alt']