import numpy as np
import time

class KalmanFilter:
    # Initialisierung von Kalman Filter
    def __init__(self, q_err,R_k):
        self.stat_k=np.array([[7],[12],[-5]])                                 # S[k]   S[0]=distance,S[1]=Velocity 
        self.pred_err_k=np.identity(3)*5000000                                 # P[k]
        self.pred_err_k_pred=np.identity(3)
        self.Phi=np.array([[1, 0.01, 0.00005],[0, 1, 0.01],[0, 0, 1]])       # Phi[k]
        self.H=np.array([[1,0,0],[0,1,0]])                                  # H[k]       
        self.mess_err=np.array([[0.00001],[0.00001]])                             #em[k]
        self.mess_err_schaetz=np.array([[0], [0], [10]])                           #es[k] [ 0, 0, eg[k-1] ]
        self.q_err=q_err#np.array([[0,0,0],[0,0,0],[0,0,100]])#np.dot(self.mess_err_schaetz,np.transpose(self.mess_err_schaetz)) #np.identity(3)                                           # Q[k]       
        self.R_k=R_k#np.array([[1,0],[0,20]]) #np.dot(self.mess_err,np.transpose(self.mess_err)) #np.identity(2)*2

    # Diese Funktion nimmt die Messwerten und gibt 
    # das Ergebnis des Kalman Filters zurück
    #
    # Bitte hier geeignete Eingabe- und Rückgabeparametern ergänzen
    def Step(self, input_k):
        mess_val=np.array([[input_k[0]],[input_k[1]]])    	               # m[k] Messwerte aus input_k     
        #print("Stat_k start",self.stat_k)
        stat_k_pred = np.dot(self.Phi,self.stat_k)              # S pred [k]
        
        #stat_k_pred=self.stat_k
        
        
        self.pred_err_k_pred=np.dot(np.dot(self.Phi,self.pred_err_k),self.Phi.transpose())+self.q_err

        # print("R_K:",self.R_k) 
        # print("H:",self.H)
        # print("pred_err_k_pred:",self.pred_err_k_pred)
        #Kalman-Verstärkung
        divident=(np.dot(self.pred_err_k_pred,self.H.T))
        try:
            divisor=np.linalg.inv(self.R_k+np.dot(np.dot(self.H,self.pred_err_k_pred),np.transpose(self.H)))        
        except np.linalg.LinAlgError:
            self.pred_err_k_pred = np.identity(3)
            divisor=np.linalg.inv(self.R_k+np.dot(np.dot(self.H,self.pred_err_k_pred),np.transpose(self.H)))
            print("LINALG ALARM")


        kal_M=np.dot(divident,divisor)
        
        self.pred_err_k=np.dot((np.identity(3)-np.dot(kal_M,self.H)),self.pred_err_k_pred)


        self.mess_err=mess_val-np.dot(self.H,self.stat_k)                   # em[k]
        #print("Stat_k Mitte",self.stat_k)
#        print("DOT",kal_M,self.mess_err)
        self.stat_k=stat_k_pred+np.dot(kal_M,self.mess_err)
        

#        print("Stat_k End",self.stat_k)
        #time.sleep(1)
        self.mess_err_schaetz=self.stat_k-stat_k_pred
        self.q_err =  np.dot(self.mess_err_schaetz,self.mess_err_schaetz.transpose())
  
        
        return self.stat_k[0],self.stat_k[1]

