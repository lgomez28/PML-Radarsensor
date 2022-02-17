import numpy as np
from collections import deque
import random

class KalmanFilter:
    # Initialisierung von Kalman Filter
    def __init__(self):
        self.stat_k=np.array([[8],[2],[10]])                                 # S[k]   S[0]=distance,S[1]=Velocity 
        self.pred_err_k=np.identity(3)*5000000                                 # P[k]
        self.pred_err_k_pred=np.identity(3)
        self.Phi=np.array([[1, 0.01, 0.00005],[0, 1, 0.01],[0, 0, 1]])       # Phi[k]
        self.H=np.array([[1,0,0],[0,1,0]])                                  # H[k]       
        self.mess_err=np.array([[10],[5]])                             #em[k]
        self.mess_err_schaetz=np.array([[0], [0], [10]])                           #es[k] [ 0, 0, eg[k-1] ]
        self.q_err=np.array([[0,0,0],[0,0,0],[0,0,593.05]])#np.dot(self.mess_err_schaetz,np.transpose(self.mess_err_schaetz)) #np.identity(3)                                           # Q[k]       
        self.R_k=np.array([[-0.80999,0],[0,4.669999]]) #np.dot(self.mess_err,np.transpose(self.mess_err)) #np.identity(2)*2
        self.init_dict={"Sinus":[40,0.2,1],
                        "Static":[1081.33,-93.85,120.82],
                        "Triangle":[593.05,-0.81,4.67],
                        "ConstantAcceleration":[1081.33,-93.85,120.82],
                        "ConstantVelocity":[885.818,-72.42,25.99]
                        }
        self.movement_list=["Sinus", "Static", "Triangle", "ConstantAcceleration", "ConstantVelocity"]
        self.movement_state="Triangle"
        self.dq_error=deque([0,0,0,0,0],5)

    # Diese Funktion nimmt die Messwerten und gibt 
    # das Ergebnis des Kalman Filters zurück
    #
    # Bitte hier geeignete Eingabe- und Rückgabeparametern ergänzen
    def Step(self, input_k):
        mess_val=np.array([[input_k[0]],[input_k[1]]])    	               # m[k] Messwerte aus input_k 
        
        stat_k_pred=np.dot(self.Phi,self.stat_k) #+self.mess_err_schaetz                # S pred [k]
        
        
        
        self.pred_err_k_pred=np.dot(np.dot(self.Phi,self.pred_err_k),self.Phi.transpose())+self.q_err

        
        #Kalman-Verstärkung
        divident=(np.dot(self.pred_err_k_pred,self.H.T))
        divisor=np.linalg.pinv((self.R_k+np.dot(np.dot(self.H,self.pred_err_k_pred),np.transpose(self.H))))                      

            
        kal_M=np.dot(divident,divisor)
        
        self.pred_err_k=np.dot((np.identity(3)-np.dot(kal_M,self.H)),self.pred_err_k_pred)


        self.mess_err=mess_val-np.dot(self.H,self.stat_k)                   # em[k]

        self.stat_k=stat_k_pred + np.dot(kal_M,self.mess_err)
        


        self.mess_err_schaetz=self.stat_k-stat_k_pred
        self.dq_error.append(np.absolute(self.mess_err))
        #print(self.mess_err_schaetz)
        # print(np.mean(np.mean(np.array(list(self.dq_error)))))
        if float(np.mean(np.mean(np.array(list(self.dq_error))))) > 50:
            new_movement_state=self.movement_list[random.randint(0,4)]
            while new_movement_state == self.movement_state:
                new_movement_state=self.movement_list[random.randint(0,4)]
            new_q= self.init_dict[new_movement_state][0]
            new_r1=self.init_dict[new_movement_state][1]
            new_r2=self.init_dict[new_movement_state][2]
            self.q_err= np.array([[0,0,0],[0,0,0],[0,0,new_q]])
            self.R_k=np.array([[new_r1,0],[0,new_r2]])
            self.movement_state=new_movement_state
            self.dq_error=deque([0,0,0,0,0],5)
            print(self.movement_state)

        
        return self.stat_k[0],self.stat_k[1]

