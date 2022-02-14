import matplotlib.pyplot as plt
import numpy as np
from DataGenerationRadar1D import GenerateData
from KalmanFilter import KalmanFilter

dist_list=[]
vel_list=[]
error_dist_list=[]
error_vel_list=[]

opt = {
        "initialDistance": 8,
        "stopTime": 1,
        "movementRange": 1,
        "frequency": 2,
        "SporadicError": 3
    }

timeAxis_test, distValues_test, velValues_test, truthDistValues_test, truthVelValues_test = GenerateData(type="Sinus", options=opt)

plt.figure()
plt.plot(timeAxis_test, distValues_test)
plt.plot(timeAxis_test, velValues_test)
plt.plot(timeAxis_test, truthDistValues_test)
plt.plot(timeAxis_test, truthVelValues_test)
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
def isNan(num):
    return num!= num
def recursive_optimation(best_error, q3, r1, r2, recursive_depth, filter_input):
    recursive_multiplier = pow(10,recursive_depth)
    step_size = 1 / recursive_multiplier
    best_value_pair = [q3, r1, r2]
    last_plotable_error = best_error
    for i in np.arange(q3 - 100 / recursive_multiplier, q3 + 100 / recursive_multiplier, 5/recursive_multiplier):
        q_err=np.array([[0,0,0],[0,0,0],[0,0,i]])
        print("Rekursive Tiefe:"+str(recursive_depth)+"\nBeste Werte für q3, r1, r2",str(best_value_pair)+"\nUpperLoopProgress"+str(i))
        for j in np.arange(r1 - 100 / recursive_multiplier, r1 + 100 / recursive_multiplier, 5/recursive_multiplier):
            for k in np.arange(r2 - 100 / recursive_multiplier, r2 + 100 / recursive_multiplier, 5/recursive_multiplier):
                R_k=np.array([[j,0],[0,k]])
                kFilter = KalmanFilter(q_err,R_k)
                error_sum = 0
                for timeAxis, distValues, velValues, truthDistValues, truthVelValues in filter_input:
                    dist_list = []
                    vel_list= []                    
                    for g in range(np.size(timeAxis)):
                        input_k=(distValues[g],velValues[g],truthDistValues[g],truthVelValues[g]) 
                        dist, vel = kFilter.Step(input_k)
                        #if isNan(dist) or isNan(vel):
                        # print(input_k,dist,vel)
                        # print("_________")
                        dist_list.append(dist[0])
                        vel_list.append(vel[0])
                        #print(truthDistValues-dist_list)
                    error_dist = np.mean(np.square(np.array(truthDistValues) - np.array(dist_list)))
                    error_vel = np.mean(np.square(np.array(truthVelValues) - np.array(vel_list)))
                    # print(np.mean(np.square(np.array(velValues-vel_list))))
                    error_sum = error_dist + error_vel
                    # print("Error Parts:", error_dist, error_vel)
                    # print("ERROR SUM:", error_sum)
                if error_sum < best_error:
                    best_error = error_sum
                    if (last_plotable_error - best_error) / best_error >= 0.001:
                        last_plotable_error = best_error
                        # print("TEst Vel:", vel_list[0], truthVelValues[0])
                        # print("TEst Dist:", dist_list[0], truthDistValues[0])
                        # print("TEst Vel ENDE:", vel_list[-1], truthVelValues[-1])
                        # print("TEst Dist EMDE:", dist_list[-1], truthDistValues[-1])
                        plt.figure()
                        plt.plot(timeAxis, dist_list)
                        plt.plot(timeAxis, vel_list)
                        plt.plot(timeAxis, truthDistValues)
                        plt.plot(timeAxis, truthVelValues)
                        plt.plot(timeAxis, (np.array(truthDistValues) - np.array(dist_list)))
                        plt.plot(timeAxis, (np.array(truthVelValues) - np.array(vel_list)))
                        plt.xlabel("time in s")
                        plt.legend(["Distance", "Velocity", "Truth distance", "Truth velocity", "Error_Dist", "Error_Vel"])
                        plt.title("Measurement Data of a 1D Radar Sensor")
                        plt.grid(True)
                        plt.show()
                    best_value_pair = [i, j, k]
                    print("NEW BEST VALUES:", best_error,best_value_pair)
    recursive_depth = recursive_depth + 1
    return recursive_optimation(best_error, best_value_pair[0], best_value_pair[1], best_value_pair[2], recursive_depth, filter_input)
    
    
    
kalman_input = []
kalman_input_data = GenerateData(type="Sinus", options=opt)
kalman_input.append(kalman_input_data)
q3_best, r1_best,r2_best = recursive_optimation(9999999999999999,0,0,0,0,kalman_input)

# best_error = 999999999999
# best_value_pair = []
# for i in range(1000):
#     q3=(i+1)/10.0
#     q_err=np.array([[0,0,0],[0,0,0],[0,0,q3]])                                           # Q[k]       
#     for j in range(1000):
#         r1=(j+1)/10.0
#         for k in range(1000):
#             r2=(k+1)/10
#             R_k=np.array([[r1,0],[0,r2]]) 
#             # Hier Ihr Kalman-Filter initialisieren
#             kFilter = KalmanFilter(q_err,R_k)
#             dist_list=[]
#             vel_list=[]

            
#             for i in range(np.size(timeAxis)):
#                 # hier die Daten ins Kalman-Filter eingeben
#                 input_k=(distValues[i],velValues[i],truthDistValues[i],truthVelValues[i]) 
#                 dist,vel = kFilter.Step(input_k)
            
#                 dist_list.append(dist)
#                 vel_list.append(vel)
#             error_dist=np.sum(np.square(truthDistValues-dist_list))
#             error_vel=np.sum(np.square(truthVelValues-vel_list))
#             err_sum = error_dist+error_vel
#             if err_sum < best_error:
#                 best_error = err_sum
#                 best_value_pair = [q3,r1,r2]
#                 print(best_error,best_value_pair)
                


# Hier das Ergebnis über die Zeit plotten.

# Um wie viel hat sich die Messgenauigkeit verbessert?
# Wie beeinflussen die Schätzung der Kovarianzmatrix Q und R die Genauigkeit
# Fügen Sie zufällige Messfehler mit der Parameter "SporadicError" hinzu, wie verhält sich das Kalman Filter?