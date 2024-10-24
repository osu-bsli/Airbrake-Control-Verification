import csv
import pandas as pd
import math
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import filterpy
from timeit import default_timer as timer

fileName = "./data/flight_data_2.csv"
timeColName = "time"
veloColName = "baro_height"
altColName = "baro_height"
accxColName = "bmx_x_accel"
accyColName = "bmx_y_accel"
acczColName = "bmx_z_accel"
gyroxColName = "bmx_x_gyro"
gyroyColName = "bmx_y_gyro"
gyrozColName = "bmx_z_gyro"

time = pd.DataFrame()
velo = pd.DataFrame()
alt = pd.DataFrame()
accx = pd.DataFrame()
accy = pd.DataFrame()
accz = pd.DataFrame()
gyrox = pd.DataFrame()
gyroy = pd.DataFrame()
gyroz = pd.DataFrame() 


m = 15
Cd_true = .333
Cd = .333
A = math.pi *(.155/2)**2
A_true = math.pi *(.155/2)**2
g = 9.81
a=0
v=0
rho=0
vterm=0
theta = 1.5

# Number of data points to use to smooth velocity (i - smoothingFactorBack : i + smoothingFactorForward)
smoothingFactorBack = 1
smoothingFactorForward = 1

data = pd.read_csv(fileName)
time = data[timeColName]
velo = data[veloColName]
alt = data[altColName]
accx = data[accxColName]
accy = data[accyColName]
accz = data[acczColName]
gyrox = data[gyroxColName]
gyroy = data[gyroyColName]
gyroz = data[gyrozColName]
trueApo = alt.max()/3.281

# Old (incorrect) model. Here for reference
def vDotSansSecant(m, rho, Cd, A, vy):
    return -g-(1/(2*m))*rho*Cd*A*vy**2

# NASA's air density model for troposphere
def airDensity(a):
    return (2116*((59+459.7-.00356*(a*3.281))/518.6)**5.256)/(1718*(59+459.7-.00356*(a*3.281)))*515.4

# Current model: (m*dv/dt = -Fd*cos0 - Fg)
def vDotSecant(m, rho, Cd, A, vy, theta):
    return -g-(1/(2*m))*rho*Cd*A*vy**2*(1/math.sin(theta))
# Current model: (v*d0/dt = g*sin0) [Vector derivation]
def thetaDot(theta, vy):
    return (g*math.sin(theta)*math.cos(theta))/vy

# Euler method for calculating current model
def predictApogeeSecantEuler(v, theta, a, time, timestep):
    vI = v
    aI = a
    thetaI = theta
    timeI = time
    while vI > 0:
        rhoI = airDensity(aI)
        dV = vDotSecant(m, rhoI, Cd, A, vI, thetaI)
        dTheta = thetaDot(thetaI, vI)

        aI += vI*timestep
        vI += dV*timestep
        thetaI += dTheta*timestep

        timeI += timestep
    theta = theta + thetaDot(theta, v)*timestep
    return theta, aI

# Runge-Kutta method for calculating current model:
#   v_{n+1} = v_n + timestep*(k_1 + 2*k_2 + 2*k_3 + k_4)/6
#       k_1 = f(t_n, v_n)
#       k_2 = f(t_n + timestep/2, v_n + timestep*k_1/2)
#       k_3 = f(t_n + timestep/2, v_n + timestep*k_2/2)
#       k_4 = f(t_n + timstep, v_n = timestep*k_3)
def predictApogeeSecantRK4(v, theta, a, time, timestep):
    vI = v
    thetaI = theta
    aI = a
    timeI = time
    changedTheta = False
    while vI > 0:
        rho1 = airDensity(aI)
        dV1 = vDotSecant(m, rho1, Cd, A, vI, thetaI)
        dTheta1 = thetaDot(thetaI, vI)

        rho2 = airDensity(aI + vI*timestep/2)
        dV2 = vDotSecant(m, rho2, Cd, A, vI + dV1*timestep/2, thetaI + dTheta1*timestep/2)
        dTheta2 = thetaDot(thetaI + dTheta1*timestep/2, vI + dV1*timestep/2)

        rho3 = airDensity(aI + vI*timestep/2)
        dV3 = vDotSecant(m, rho3, Cd, A, vI + dV2*timestep/2, thetaI + dTheta2*timestep/2)
        dTheta3 = thetaDot(thetaI + dTheta2*timestep/2, vI + dV2*timestep/2)

        rho4 = airDensity(aI + vI*timestep)
        dV4 = vDotSecant(m, rho4, Cd, A, vI + dV3*timestep, thetaI + dTheta3*timestep)
        dTheta4 = thetaDot(thetaI + dTheta3*timestep, vI + dV3*timestep)

        vI += (dV1 + 2*dV2 + 2*dV3 + dV4)*timestep/6
        thetaI += (dTheta1 + 2*dTheta2 + 2*dTheta3 + dTheta4)*timestep/6
        if (changedTheta == False):
            theta += (dTheta1 + 2*dTheta2 + 2*dTheta3 + dTheta4)*timestep/6
            changedTheta = True
        aI += vI*timestep
        timeI += timestep
    return theta, aI

# Analytic solution for old model
def predictApogeeSansSecantAnalytic(m, Cd, A, v, a, timestep):
    rho = (2116*((59+459.7-.00356*(a*3.281))/518.6)**5.256)/(1718*(59+459.7-.00356*(a*3.281)))*515.4
    return ((-m)/(rho*Cd*A))*math.log(((rho*Cd*A)/(2*m*g))*v**2+1)+v*math.sqrt((2*m)/(g*rho*Cd*A))*math.atan(v*math.sqrt((rho*Cd*A)/(2*m*g)))+a

# Runge-Kutta model for calculating old model
def predictApogeeSansSecantRK4(v, a, time, timestep):
    vI = v
    aI = a
    timeI = time
    while vI > 0:
        rho1 = airDensity(aI)
        dV1 = vDotSansSecant(m, rho1, Cd, A, vI)

        rho2 = airDensity(aI + vI*timestep/2)
        dV2 = vDotSansSecant(m, rho2, Cd, A, vI + dV1*timestep/2)

        rho3 = airDensity(aI + vI*timestep/2)
        dV3 = vDotSansSecant(m, rho3, Cd, A, vI + dV2*timestep/2)

        rho4 = airDensity(aI + vI*timestep)
        dV4 = vDotSansSecant(m, rho4, Cd, A, vI + dV3*timestep)

        vI += (dV1 + 2*dV2 + 2*dV3 + dV4)*timestep/6
        aI += vI*timestep
        timeI += timestep
    return aI



# Find the index where the air brakes will start
startIndx = -1
endIndx = -1
for i in range(400, 900):
    v, b = np.polyfit(time.iloc[i - smoothingFactorBack : i + smoothingFactorForward],alt.iloc[i - smoothingFactorBack : i + smoothingFactorForward],1) / 3.281
    #v = velo.iloc[i]
    if (v < 248 and startIndx == -1):
        startIndx = i
    if (alt.iloc[i] == alt.max() and endIndx == -1):
        endIndx = i



file = open("out.csv", 'w', newline='')
out = csv.writer(file)
out.writerow(["time", "altitude", "10 prediction", "10 error", "10 time", "1 prediction", "1 error", "1 time", "0.1 prediction", "0.1 error", "0.1 time", "0.01 prediction", "0.01 error", "0.01 time", "0.001 prediction", "0.001 error", "0.001 time", "0.0001 prediction", "0.0001 error", "0.0001 time"])

apogee = (alt.max())/3.281

times = []
rk4apos = [[],[],[],[],[],[]]
rk4errs = [[],[],[],[],[],[]]
rk4tims = [[],[],[],[],[],[]]
timesteps = [10,1,0.1,0.01,0.001,0.0001]

# Run apogee predicitions
for i in range(startIndx, endIndx):
    # Get rocket state
    a = alt.iloc[i]/3.281
    v, b = np.polyfit(time.iloc[i - smoothingFactorBack : i + smoothingFactorForward],alt.iloc[i - smoothingFactorBack : i + smoothingFactorForward],1) / 3.281

    times.append(time.iloc[i])
    # ~~~~~~ Experiments ~~~~~~~~
    for j in range(0,6):
        start = timer()
        x , pred = predictApogeeSecantRK4(v, theta, a, time.iloc[i], timesteps[j])
        end = timer()
        rk4apos[j].append(pred) 
        rk4errs[j].append(pred-apogee)
        rk4tims[j].append(end-start)
    theta -= thetaDot(theta, v)
    if theta > 1.5 or theta < 0.0001:
        theta = 0.0001

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(time.iloc[i], v, theta, rk4apos[0][-1])


    out.writerow([time.iloc[i], a, rk4apos[0][-1], rk4errs[0][-1], rk4tims[0][-1], rk4apos[1][-1], rk4errs[1][-1], rk4tims[1][-1], rk4apos[2][-1], rk4errs[2][-1], rk4tims[2][-1], rk4apos[3][-1], rk4errs[3][-1], rk4tims[3][-1], rk4apos[4][-1], rk4errs[4][-1], rk4tims[4][-1], rk4apos[5][-1], rk4errs[5][-1], rk4tims[5][-1]])

fig, ax = plt.subplots(2,3)
for r in range(0,2):
    for c in range(0,3):
        ax[r,c].plot(times, rk4apos[3*r+c])
        ax[r,c].set_title("Timestep = " + str(timesteps[3*r+c]))
plt.show()
fig1, ax1 = plt.subplots(2,3)
for r in range(0,2):
    for c in range(0,3):
        ax1[r,c].plot(times, rk4errs[3*r+c])
        ax1[r,c].set_title("Timestep = " + str(timesteps[3*r+c]))
plt.show()
fig2, ax2 = plt.subplots(2,3)
for r in range(0,2):
    for c in range(0,3):
        ax2[r,c].plot(times, rk4tims[3*r+c])
        ax2[r,c].set_title("Timestep = " + str(timesteps[3*r+c]))
plt.show()


