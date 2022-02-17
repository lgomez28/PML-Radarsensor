import matplotlib.pyplot as plt
import numpy as np
from DataGenerationRadar1D import GenerateData
from KalmanFilter import KalmanFilter
import random

dist_list=[]
vel_list=[]

error_vel=[]
error_dist=[]

opt = {
        "initialDistance": 8,
        "stopTime": 1,
        "movementRange": 1,
        "frequency": 2,
        "SporadicError": 5,
        "initialVelocity": 2,
        "acceleration": 2,
        "velocity":5
        
    }

movementtype=["Triangle","Sinus","ConstantVelocity","ConstantAcceleration","Static"]

timeAxis=[]
distValues=[]
velValues=[]
truthDistValues=[]
truthVelValues=[]


for i in range(10):
    temp_timeAxis, temp_distValues, temp_velValues, temp_truthDistValues, temp_truthVelValues = GenerateData(type=movementtype[random.randint(0,4)], options=opt)
    for j in range(0,len(temp_timeAxis)):
        timeAxis.append(temp_timeAxis[j]+ i)
        distValues.append(temp_distValues[j])
        velValues.append(temp_velValues[j])
        truthDistValues.append(temp_truthDistValues[j])
        truthVelValues.append(temp_truthVelValues[j])
        if j==100 or j==0:
            # print(distValues[j])
            # print(velValues[j])
            if distValues[j] >20 or distValues[j] < -20:
                distValues[j] = distValues[j-1]
            if velValues[j] >20 or velValues[j] < -20:
                velValues[j] = velValues[j-1]
plt.figure()
ax = plt.gca()
ax.set_ylim([-50,50])
plt.plot(timeAxis, distValues)
plt.plot(timeAxis, velValues)
plt.plot(timeAxis, truthDistValues)
plt.plot(timeAxis, truthVelValues)
plt.xlabel("time in s")
plt.legend(["Distance", "Velocity", "Truth distance", "Truth velocity"])
plt.title("Measurement Data of a 1D Radar Sensor")
plt.grid(True)
plt.show()

'''
Aufgabe:
1. Implementieren Sie ein Kalman-Filter, das die Messdaten als Eingangsdaten nimmt.
2. Testen Sie das Kalman-Filter mit verschiedener Objektbewegungsarten.
'''



kFilter = KalmanFilter()
            
for i in range(np.size(timeAxis)):
    # hier die Daten ins Kalman-Filter eingeben
    input_k=(distValues[i],velValues[i],truthDistValues[i],truthVelValues[i]) 
    dist,vel = kFilter.Step(input_k)
    dist_list.append(dist[0])
    vel_list.append(vel[0])
            #     error_dist.append(np.power(truthDistValues[i]-dist,2))
            #     error_vel.append(np.power(truthVelValues[i]-vel,2))
            #     dist_list.append(dist)
            #     vel_list.append(vel)
            # # error_dist=np.sum(np.square(truthDistValues-dist_list))
            # # error_vel=np.sum(np.square(truthVelValues-vel_list))
            # err_sum = np.mean(np.array(error_dist)+np.array(error_vel))
            # error_dist= []
            # error_vel=[]
            # if err_sum < best_error:
            #     best_error = err_sum
            #     best_value_pair = [q3,r1,r2]
            #     print(best_error,best_value_pair)


dist_mess_err = np.square((np.subtract(np.array(truthDistValues),np.array(distValues))))
vel_mess_err = np.square((np.subtract(np.array(truthVelValues),np.array(velValues))))

dist_kal_err = np.square((np.subtract(np.array(truthDistValues),np.array(dist_list))))
vel_kal_err = np.square((np.subtract(np.array(truthVelValues),np.array(vel_list))))

# print(truthDistValues)
# print(vel_list)
# print(dist_list)

print(np.mean(dist_mess_err))
print(np.mean(vel_mess_err))
print(np.mean(dist_kal_err))
print(np.mean(vel_kal_err))


plt.figure()
ax = plt.gca()
ax.set_ylim([-50,50])

plt.plot(timeAxis, dist_list)
plt.plot(timeAxis, vel_list)
plt.plot(timeAxis, truthDistValues)
plt.plot(timeAxis, truthVelValues)


plt.xlabel("time in s")
plt.legend(["Distance", "Velocity", "Truth distance", "Truth velocity"])
plt.title("Measurement Data of a 1D Radar Sensor")
plt.grid(True)
plt.show()


plt.figure()
plt.plot(timeAxis, dist_mess_err)
plt.plot(timeAxis, vel_mess_err)
plt.xlabel("time in s")
plt.legend(["Distance Mess Error", "Velocity Mess Error"])
plt.title("Measurement Data of a 1D Radar Sensor")
plt.grid(True)
plt.show()



plt.figure()
plt.plot(timeAxis, dist_kal_err)
plt.plot(timeAxis, vel_kal_err)
plt.xlabel("time in s")
plt.legend([ "Kalman Error distance", "Kalman Error velocity"])
plt.title("Measurement Data of a 1D Radar Sensor")
plt.grid(True)
plt.show()

        # Hier das Ergebnis über die Zeit plotten.

# Um wie viel hat sich die Messgenauigkeit verbessert?
# Wie beeinflussen die Schätzung der Kovarianzmatrix Q und R die Genauigkeit
# Fügen Sie zufällige Messfehler mit der Parameter "SporadicError" hinzu, wie verhält sich das Kalman Filter?