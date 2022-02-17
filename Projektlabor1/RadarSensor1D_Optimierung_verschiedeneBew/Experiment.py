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
        "SporadicError": 5
    }

'''
Aufgabe:
1. Implementieren Sie ein Kalman-Filter, das die Messdaten als Eingangsdaten nimmt.
2. Testen Sie das Kalman-Filter mit verschiedener Objektbewegungsarten.
'''
def isNan(num):
    return num!= num
def recursive_optimation(best_error, q3, r1, r2, recursive_depth, filter_input, max_recursion_depth):
    recursive_multiplier = pow(10,recursive_depth)
    step_size = 1 / recursive_multiplier
    best_value_pair = [q3, r1, r2]
    last_plotable_error = best_error
    for i in np.arange(q3 - 1000 / recursive_multiplier, q3 + 1000 / recursive_multiplier, 25/recursive_multiplier):
        q_err=np.array([[0,0,0],[0,0,0],[0,0,i]])
        print("Rekursive Tiefe:"+str(recursive_depth)+"\nBeste Werte für q3, r1, r2",str(best_value_pair)+"\nUpperLoopProgress"+str(i))
        for j in np.arange(r1 - 200 / recursive_multiplier, r1 + 200 / recursive_multiplier, 10/recursive_multiplier):
            for k in np.arange(r2 - 200 / recursive_multiplier, r2 + 200 / recursive_multiplier, 10/recursive_multiplier):
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
                        # plt.figure()
                        # plt.plot(timeAxis, dist_list)
                        # plt.plot(timeAxis, vel_list)
                        # plt.plot(timeAxis, truthDistValues)
                        # plt.plot(timeAxis, truthVelValues)
                        # plt.plot(timeAxis, np.square(np.array(truthDistValues) - np.array(dist_list)))
                        # plt.plot(timeAxis, np.square(np.array(truthVelValues) - np.array(vel_list)))
                        # plt.xlabel("time in s")
                        # plt.legend(["Distance", "Velocity", "Truth distance", "Truth velocity", "Error_Dist", "Error_Vel"])
                        # plt.title("Measurement Data of a 1D Radar Sensor")
                        # plt.grid(True)
                        # plt.show()
                    best_value_pair = [i, j, k]
                    print("NEW BEST VALUES:", best_error,best_value_pair)
    recursive_depth = recursive_depth + 1
    if recursive_depth  > max_recursion_depth:
        return best_value_pair[0], best_value_pair[1], best_value_pair[2]
    else:
        return recursive_optimation(best_error, best_value_pair[0], best_value_pair[1], best_value_pair[2], recursive_depth, filter_input, max_recursion_depth)
    
    
    
kalman_input = []
for i in range(0,5):
        kalman_input_data = GenerateData(type="Sinus", options=opt)
        kalman_input.append(kalman_input_data)
q3_best_sinus, r1_best_sinus,r2_best_sinus = recursive_optimation(9999999999999999,0,0,0,0,kalman_input, max_recursion_depth = 4)
print("Best Values for Sinus:\nQ3:",q3_best_sinus,"\nr1:",r1_best_sinus,"\nr2:", r2_best_sinus )

f= open("Sinus.txt","w+")
f.write("Best Values for Sinus:\nQ3:"+str(q3_best_sinus)+"\nr1:"+str(r1_best_sinus)+"\nr2:"+str(r2_best_sinus))
f.close() 
           


kalman_input = []
for i in range(0,5):
        kalman_input_data = GenerateData(type="Triangle", options=opt)
        kalman_input.append(kalman_input_data)
q3_best_triangle, r1_best_triangle, r2_best_triangle = recursive_optimation(9999999999999999,0,0,0,0,kalman_input, max_recursion_depth = 4)
print("Best Values for Triangle:\nQ3:",q3_best_triangle,"\nr1:",r1_best_triangle,"\nr2:", r2_best_triangle )


f= open("Triangle.txt","w+")
f.write("Best Values for Triangle:\nQ3:"+str(q3_best_triangle)+"\nr1:"+str(r1_best_triangle)+"\nr2:"+str(r2_best_triangle))
f.close() 

kalman_input = []
opt = {
        "initialDistance": 8,
        "stopTime": 1,
        "movementRange": 1,
        "frequency": 2,
        "SporadicError": 5,
        "initialVelocity":2,
        "acceleration":1
    }
for i in range(0,5):
        kalman_input_data = GenerateData(type="ConstantAcceleration", options=opt)
        kalman_input.append(kalman_input_data)
q3_best_ConstantAcceleration, r1_best_ConstantAcceleration,r2_best_ConstantAcceleration = recursive_optimation(9999999999999999,0,0,0,0,kalman_input, max_recursion_depth = 4)
print("Best Values for ConstantAcceleration:\nQ3:",q3_best_ConstantAcceleration,"\nr1:",r1_best_ConstantAcceleration,"\nr2:", r2_best_ConstantAcceleration )


f= open("ConstantAcceleration.txt","w+")
f.write("Best Values for ConstantAcceleration:\nQ3:"+str(q3_best_ConstantAcceleration)+"\nr1:"+str(r1_best_ConstantAcceleration)+"\nr2:"+str(r2_best_ConstantAcceleration))
f.close() 

opt = {
        "initialDistance": 8,
        "stopTime": 1,
        "movementRange": 1,
        "frequency": 2,
        "SporadicError": 5,
        "initialVelocity":2,
        "velocity":1
    }
kalman_input = []
for i in range(0,5):
        kalman_input_data = GenerateData(type="ConstantVelocity", options=opt)
        kalman_input.append(kalman_input_data)
q3_best_ConstantVelocity, r1_best_ConstantVelocity,r2_best_ConstantVelocity = recursive_optimation(9999999999999999,0,0,0,0,kalman_input, max_recursion_depth = 4)
print("Best Values for ConstantVelocity:\nQ3:",q3_best_ConstantVelocity,"\nr1:",r1_best_ConstantVelocity,"\nr2:", r2_best_ConstantVelocity )

f= open("ConstantVelocity.txt","w+")
f.write("Best Values for ConstantVelocity:\nQ3:"+str(q3_best_ConstantVelocity)+"\nr1:"+str(r1_best_ConstantVelocity)+"\nr2:"+str(r2_best_ConstantVelocity))
f.close() 

kalman_input = []
for i in range(0,5):
        kalman_input_data = GenerateData(type="Static", options=opt)
        kalman_input.append(kalman_input_data)
q3_best_Static, r1_best_Static,r2_best_Static = recursive_optimation(9999999999999999,0,0,0,0,kalman_input, max_recursion_depth = 4)
print("Best Values for ConstantVelocity:\nQ3:",q3_best_Static,"\nr1:",r1_best_Static,"\nr2:", r2_best_Static )

f= open("Static.txt","w+")
f.write("Best Values for Static:\nQ3:"+str(q3_best_Static)+"\nr1:"+str(r1_best_Static)+"\nr2:"+str(r2_best_Static))
f.close() 

print("Best Values for Sinus:\nQ3:",q3_best_sinus,"\nr1:",r1_best_sinus,"\nr2:", r2_best_sinus )
print("Best Values for Triangle:\nQ3:",q3_best_triangle,"\nr1:",r1_best_triangle,"\nr2:", r2_best_triangle )
print("Best Values for ConstantAcceleration:\nQ3:",q3_best_ConstantAcceleration,"\nr1:",r1_best_ConstantAcceleration,"\nr2:", r2_best_ConstantAcceleration )
print("Best Values for ConstantVelocity:\nQ3:",q3_best_ConstantVelocity,"\nr1:",r1_best_ConstantVelocity,"\nr2:", r2_best_ConstantVelocity )
print("Best Values for Static:\nQ3:",q3_best_Static,"\nr1:",r1_best_Static,"\nr2:", r2_best_Static)


# Hier das Ergebnis über die Zeit plotten.

# Um wie viel hat sich die Messgenauigkeit verbessert?
# Wie beeinflussen die Schätzung der Kovarianzmatrix Q und R die Genauigkeit
# Fügen Sie zufällige Messfehler mit der Parameter "SporadicError" hinzu, wie verhält sich das Kalman Filter?