import csv
import pandas as pd
import math
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import filterpy

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
Cd = .333
A = math.pi *(.155/2)**2
g = 9.81
a=0
v=0
rho=0
vterm=0
timestep = .01
theta = math.pi/6

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
    return -g-(1/(2*m))*rho*Cd*A*vy**2*(1/math.cos(theta))
# Current model: (v*d0/dt = g*sin0) [Vector derivation]
def thetaDot(theta, vy):
    return (g*math.sin(theta)*math.cos(theta))/vy

# Euler method for calculating current model
def predictApogeeSecantEuler(v, theta, a, time):
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
def predictApogeeSecantRK4(v, theta, a, time):
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
def predictApogeeSansSecantAnalytic(m, Cd, A, v, a):
    rho = (2116*((59+459.7-.00356*(a*3.281))/518.6)**5.256)/(1718*(59+459.7-.00356*(a*3.281)))*515.4
    return ((-m)/(rho*Cd*A))*math.log(((rho*Cd*A)/(2*m*g))*v**2+1)+v*math.sqrt((2*m)/(g*rho*Cd*A))*math.atan(v*math.sqrt((rho*Cd*A)/(2*m*g)))+a

# Runge-Kutta model for calculating old model
def predictApogeeSansSecantRK4(v, a, time):
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




startIndx = -1
endIndx = -1
for i in range(400, 900):
    v, b = np.polyfit(time.iloc[i - smoothingFactorBack : i + smoothingFactorForward],alt.iloc[i - smoothingFactorBack : i + smoothingFactorForward],1) / 3.281
    #v = velo.iloc[i]
    if (v < 248 and startIndx == -1):
        startIndx = i
    if (alt.iloc[i] == alt.max() and endIndx == -1):
        endIndx = i
thetaSecEuler = theta
thetaSecRK4 = theta
perErrorSansAnalytic = []
perErrorSecEuler = []
perErrorSecRK4 = []
perErrorSansRK4 = []
apoE = []
apoSA = []
apoSR = []
apoR = []
times = []
alts = []

file = open("predictions_100_Derivative.csv", 'w', newline='')
out = csv.writer(file)
out.writerow(["Time", "Altitude", "Euler Prediction", "Euler Error", "RK4 Prediction", "RK4 Error"])


velo_alpha = 1

v, b = np.polyfit(time.iloc[startIndx-1 - smoothingFactorBack : startIndx-1 + smoothingFactorForward],alt.iloc[startIndx-1 - smoothingFactorBack : startIndx-1 + smoothingFactorForward],1) / 3.281
for i in range(startIndx, endIndx):
    a = alt.iloc[i]/3.281
    #v, b = np.polyfit(time.iloc[i - smoothingFactorBack : i + smoothingFactorForward],alt.iloc[i - smoothingFactorBack : i + smoothingFactorForward],1) / 3.281
    #v = velo.iloc[i]
    ax = accx.iloc[i]
    ay = accy.iloc[i]
    az = accz.iloc[i]
    acc = math.sqrt(ax**2 + ay**2 + az**2)


    gx = gyrox.iloc[i]
    gy = gyroy.iloc[i]
    gz = gyroz.iloc[i]

    v_alt_deriv, b = np.polyfit(time.iloc[i - smoothingFactorBack : i + smoothingFactorForward],alt.iloc[i - smoothingFactorBack : i + smoothingFactorForward],1) / 3.281
    v_acc_int = v + (acc*(time.iloc[i]-time.iloc[i-1]))*math.cos(thetaSecRK4)

    v = velo_alpha*v_alt_deriv + (1 - velo_alpha)*v_acc_int
    print(v, v_alt_deriv)

    apoSansAnalytic = predictApogeeSansSecantAnalytic(m, Cd, A, v, a)
    thetaSecEuler, apoSecEuler = predictApogeeSecantEuler(v, thetaSecEuler, a, time.iloc[i])
    thetaSecRK4, apoSecRK4 = predictApogeeSecantRK4(v, thetaSecRK4, a, time.iloc[i])
    apoSansRK4 = predictApogeeSansSecantRK4(v, a, time.iloc[i])

    perErrorSansAnalytic.append((apoSansAnalytic-(trueApo))/(trueApo)*100)
    perErrorSecEuler.append((apoSecEuler-(trueApo))/(trueApo)*100)
    perErrorSecRK4.append((apoSecRK4-(trueApo))/(trueApo)*100)
    perErrorSansRK4.append((apoSansRK4-(trueApo))/(trueApo)*100)
    apoE.append(apoSecEuler)
    apoR.append(apoSecRK4)
    apoSA.append(apoSansAnalytic)
    apoSR.append(apoSansRK4)
    times.append(time.iloc[i])
    alts.append(a)

    out.writerow([time.iloc[i], a, apoSecEuler, (apoSecEuler-(trueApo))/(trueApo)*100, apoSecRK4, (apoSecRK4-(trueApo))/(trueApo)*100])

fig, ax = plt.subplots(2,2)
ax[0,0].plot(times, apoE)
ax[0,0].set_title("Euler Prediction")
ax[0,1].plot(times, apoR)
ax[0,1].set_title("RK4 Prediction")
ax[1,0].plot(times, perErrorSecEuler)
ax[1,0].set_title("Euler Error")
ax[1,1].plot(times, perErrorSecRK4)
ax[1,1].set_title("RK4 Error")
plt.show()



