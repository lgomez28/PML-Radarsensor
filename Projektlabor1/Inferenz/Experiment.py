import matplotlib.pyplot as plt
import numpy as np
from DataGenerationRadar1D import GenerateData
from KalmanFilter import KalmanFilter

dist_list=[]
vel_list=[]

error_vel=[]
error_dist=[]

opt = {
        "initialDistance": 8,
        "stopTime": 1,
        "movementRange": 1,
        "frequency": 2,
        "SporadicError": 10
    }

timeAxis, distValues, velValues, truthDistValues, truthVelValues = GenerateData(type="Sinus", options=opt)


plt.figure()
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