import csv
import pandas as pd
import math
import numpy as np
from scipy import optimize

fileName = "data/flight_data_2.csv"
timeColName = "time"
veloColName = "velo"
altColName = "alt"

time = pd.DataFrame()
velo = pd.DataFrame()
alt = pd.DataFrame()


m = 15
C = .3
A = math.pi *(.155/2)**2
g = 9.8
a=0
v=0
rho=0
vterm=0

data = pd.read_csv(fileName)
time = data[timeColName]
#velo = data[veloColName]
alt = data[altColName]

def findC(x):
    c = .5*rho*x*A
    maxTime = (m * math.sqrt((.5*rho*C*A)/(m*g)) * math.atan(v*math.sqrt((.5*rho*C*A)/(m*g)))) / (.5*rho*C*A)
    ymax = m/(.5*rho*C*A) * math.log(math.cos(math.sqrt((.5*rho*C*A*g)/m)*maxTime)) + v*maxTime + a
    return 9144-ymax

def calcC():
    return optimize.root(findC, .5, method='anderson')

def predictApogee():
    maxTime = (m * math.sqrt((.5*rho*C*A)/(m*g)) * math.atan(v*math.sqrt((.5*rho*C*A)/(m*g)))) / (.5*rho*C*A)
    ymax = m/(.5*rho*C*A) * math.log(math.cos(math.sqrt((.5*rho*C*A*g)/m)*maxTime)) + v*maxTime + a
    return maxTime, ymax

with open("out.csv", "w", newline='') as file:
    writer = csv.writer(file)
    for i in range(120,650):
        a = alt.iloc[i] / 3.281
        #v = (alt.iloc[i] - alt.iloc[i-1]) / (time.iloc[i]-time.iloc[i-1]) / 3.281
        v, b = np.polyfit(time.iloc[i-5:i],alt.iloc[i-5:i],1) / 3.281
        #v = velo.iloc[i]
        rho = 1.225*((288.16-.0065*(.3048*a))/288.16)**4.2561
        vterm = math.sqrt(m*g/(.5*rho*C*A))

        try:
            # Start timer
            maxTime, ymax = predictApogee()
            # End Timer

            
            writer.writerow([time.iloc[i], a, v, maxTime, ymax])
        except:
            print("Domain error skipped")
        # Start Timer
        c = calcC()
        # End Timer
        print(c.x)

