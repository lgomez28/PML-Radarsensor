import numpy as np

class KalmanFilter:
    # Initialisierung von Kalman Filter
    def __init__(self):
        self.stat_k=np.array([[7],[12],[-5]])                                 # S[k]   S[0]=distance,S[1]=Velocity 
        self.pred_err_k=np.identity(3)*5000000                                 # P[k]
        self.pred_err_k_pred=np.identity(3)
        self.Phi=np.array([[1, 0.01, 0.00005],[0, 1, 0.01],[0, 0, 1]])       # Phi[k]
        self.H=np.array([[1,0,0],[0,1,0]])                                  # H[k]       
        self.mess_err=np.array([[10],[5]])                             #em[k]
        self.mess_err_schaetz=np.array([[0], [0], [100]])                           #es[k] [ 0, 0, eg[k-1] ]
        self.q_err=np.array([[0,0,0],[0,0,0],[0,0,186]])#np.dot(self.mess_err_schaetz,np.transpose(self.mess_err_schaetz)) #np.identity(3)                                           # Q[k]       
        self.R_k=np.array([[0.02,0],[0,1]]) #np.dot(self.mess_err,np.transpose(self.mess_err)) #np.identity(2)*2

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
        divisor=np.linalg.inv((self.R_k+np.dot(np.dot(self.H,self.pred_err_k_pred),np.transpose(self.H))))                      

            
        kal_M=np.dot(divident,divisor)
        
        self.pred_err_k=np.dot((np.identity(3)-np.dot(kal_M,self.H)),self.pred_err_k_pred)


        self.mess_err=mess_val-np.dot(self.H,self.stat_k)                   # em[k]

        self.stat_k=stat_k_pred+np.dot(kal_M,self.mess_err)
        


        self.mess_err_schaetz=self.stat_k-stat_k_pred
  

        return self.stat_k[0],self.stat_k[1]
