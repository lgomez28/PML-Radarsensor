import numpy as np
import time
from collections import deque
class KalmanFilter:
    # Initialisierung von Kalman Filter
    def __init__(self, qerr, rk, pred_err_multiplier ,printBool = False):
        self.stat_k=np.array([[1,1,1],[1, 1, 1], [0, 0, 0]])                                 # S[k]   S[0]=distance,S[1]=Velocity
        self.pred_err_k = np.identity(3)        *  pred_err_multiplier             # P[k]
        self.pred_err_k_pred = np.identity(3)
        self.Phi = np.array([[1, 1/30., 0.5*np.square(1/30.)],[0, 1, 1/30.], [0, 0, 1]])       # Phi[k]
        self.H = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 0]])                                  # H[k]       
        self.mess_err = np.array([[0.00001, 0, 0], [0.00001, 0, 0]])                             #em[k]
        self.mess_err_schaetz = rk                   #es[k] [ 0, 0, eg[k-1] ]
        self.q_err =  np.dot(self.mess_err_schaetz,self.mess_err_schaetz.transpose())       
        self.R_k =  rk
        self.printBool = printBool

    # Diese Funktion nimmt die Messwerten und gibt 
    # das Ergebnis des Kalman Filters zurück
    #
    # Bitte hier geeignete Eingabe- und Rückgabeparametern ergänzen
    def Step(self, input_k, input_k_past, currentvel):
        pos=np.array(input_k)
        pos_past = np.array(input_k_past)
        velocity = currentvel
        mess_val = np.array([pos,velocity, [0, 0, 0]])    	               # m[k] Messwerte aus input_k               
        stat_k_pred = np.dot(self.Phi, self.stat_k)
        self.pred_err_k_pred=np.dot(np.dot(self.Phi, self.pred_err_k), self.Phi.transpose()) + self.q_err
        #Kalman-Verstärkung
        divident = (np.dot(self.pred_err_k_pred, self.H.T))
        try:
            # print("divisor")
            divisor=np.linalg.inv(self.R_k+np.dot(np.dot(self.H,self.pred_err_k_pred),np.transpose(self.H)))        
        except np.linalg.LinAlgError:
            print("Reset of pred_err_k due to linalg error")
            self.pred_err_k_pred = np.identity(3)
            divisor=np.linalg.inv(self.R_k+np.dot(np.dot(self.H,self.pred_err_k_pred),np.transpose(self.H)))                  
        kal_M=np.dot(divident,divisor)
        
        self.pred_err_k = np.dot((np.identity(3)-np.dot(kal_M, self.H)), self.pred_err_k_pred)
        self.mess_err= mess_val - np.dot(self.H ,self.stat_k)                # em[k]

        self.stat_k=stat_k_pred+np.dot(kal_M,self.mess_err)
        


        self.mess_err_schaetz = self.stat_k-stat_k_pred
        self.q_err =  np.dot(self.mess_err_schaetz,self.mess_err_schaetz.transpose())
        # self.R_k = np.dot(self.mess_err,self.mess_err.transpose())
        # print("RK Actualized")
        # if self.printBool:
        #     print(self.stat_k[0],self.stat_k[1])
        return self.stat_k[0],self.stat_k[1]
    
  