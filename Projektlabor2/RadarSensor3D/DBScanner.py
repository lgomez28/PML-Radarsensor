# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 20:39:20 2022

@author: jgaertner
"""
import numpy as np
from collections import deque
import matplotlib.pyplot as pyplot
from PIL import Image

class DBScanner:
    def __init__(self, min_points, max_dist, num_active_points = 80, num_active_points_per_cluster = 6, delta_t = 1/30.):
        self.min_points = min_points
        self.max_dist = max_dist
        self.active_points = deque(maxlen = 2000)
        self.all_data_points = deque(maxlen = 100000)
        self.clustered_data_points = []
        self.num_active_points_per_cluster = num_active_points_per_cluster
        self.all_clusters = {}
        self.active_clusters = {}
        self.cluster_counter = 0
        self.plots = []
        self.vidoe_counter = 0
        self.delta_t = delta_t
    
    def add_data_point(self, data_point):
        data_point = data_point[0:3]
        #print("Start:",len(self.active_points))
        self.all_data_points.append(data_point)
        self.active_points.append(data_point)
        #print("Post Append",len(self.active_points))
        active_neighbors = self.getNeighbors(data_point)
        #print("Post Neighbor Search",len(self.active_points))
        found_fitting_cluster = False
        found_cluster_key = -1
        old_cluster_center = []
        new_cluster_center = []
        # Prüfen ob einer der Nachbarn schon einem Cluster zugeordent ist
        #print(data_point)
        #print(active_neighbors)
        for neighbor in active_neighbors:
            #print(neighbor)
            for cluster_key in self.active_clusters.keys():
                # print(self.active_clusters[cluster_key])
                positions = list(self.active_clusters[cluster_key])[::][0]
                # print(positions)
                if any((neighbor == ele).all() for ele in positions) and not found_fitting_cluster:
                    #if not any((new_array == elem).all() for elem in dequeList)):
                    #print("found cluster")
                    found_fitting_cluster = True
                    found_cluster_key = cluster_key
                    old_cluster_center = self.getClusterCenter(cluster_key)
                    new_cluster_center = self.getClusterCenter(found_cluster_key)
                    speed = (new_cluster_center - old_cluster_center) * 1 / self.delta_t
                    self.active_clusters[cluster_key].append([data_point, speed])
                    self.all_clusters[cluster_key].append([self.getClusterCenter(cluster_key), speed])
                    self.clustered_data_points.append(data_point)
        # Wenn kein Cluster gefunden wurde prüfen ob genug Nachbarn vorhanden sind um neues Cluster zu gründen
        if not found_fitting_cluster and len(active_neighbors) > self.min_points:
            print("new Cluster")
            new_active_cluster = deque([],maxlen = self.num_active_points_per_cluster)
            self.active_clusters[self.cluster_counter] = new_active_cluster
            found_cluster_key = self.cluster_counter
            old_cluster_center = [0,0,0]
            for neighbor in active_neighbors:
                new_cluster_center = self.getClusterCenter(found_cluster_key)
                speed = (new_cluster_center - old_cluster_center) * 1 / self.delta_t
                new_active_cluster.append([neighbor, speed])
                old_cluster_center = new_cluster_center
                self.clustered_data_points.append(neighbor)
            new_cluster_center = self.getClusterCenter(found_cluster_key)
            speed = (new_cluster_center - old_cluster_center) * 1 / self.delta_t
            new_active_cluster.append([data_point,[0,0,0]])
            self.all_clusters[self.cluster_counter] = deque(list(new_active_cluster), maxlen=10000)
            found_fitting_cluster = True
            self.cluster_counter += 1
        # Aufräumen und abgleich mit aktiven Daten
        for cluster_key in self.active_clusters.keys():
            to_remove = False
            for clusterpoint in self.active_clusters[cluster_key]:
                #print(clusterpoint)
               # print(clusterpoint[0])
                if not any((clusterpoint[0] == active_point).all()for active_point in self.active_points):
                    print("Removed old Point from active Cluster")
                    to_remove = True
            if to_remove:
                self.active_clusters[cluster_key].pop()
                    
                    
                    
        # Erstellen von Plots für Videos
        # self.vidoe_counter += 1
        # if self.vidoe_counter % 20 == 0:
        #     color_list=['c','m','r','g','b','y','darkred', 'darkolivegreen', 'gold','lime','slategray']
        #     fig2=pyplot.figure()
        #     ax2=fig2.add_subplot(111,projection='3d')
        #     for cluster_key in self.active_clusters.keys():
        #         cluster_points = self.active_clusters[cluster_key]
        #         ax2.scatter([clusterpoint[0][0] for clusterpoint in cluster_points],[clusterpoint[0][1] for clusterpoint in cluster_points],[clusterpoint[0][2] for clusterpoint in cluster_points],s=0.7,color=color_list[cluster_key])        
        #     for cluster_key in self.active_clusters.keys():
        #         clustercenter = self.getClusterCenter(cluster_key)
        #         ax2.scatter(clustercenter[0], clustercenter[1], clustercenter[2],s=25, marker = "x", color=color_list[cluster_key])
        #     for cluster_key in self.all_clusters.keys():
        #         cluster_points = self.all_clusters[cluster_key]
        #         ax2.scatter([clusterpoint[0][0] for clusterpoint in cluster_points],[clusterpoint[0][1] for clusterpoint in cluster_points],[clusterpoint[0][2] for clusterpoint in cluster_points],s=0.7,color=color_list[cluster_key])
        #     self.plots.append(self.fig2img(fig2))
        
        # Berechnung der GEschwindigkeit des aktuelle Datenpunkte über die Zentren der CLuster
        # [0,0,0] Wenn kein Cluster vorhanden ist.
        if found_fitting_cluster:
            new_cluster_center = self.getClusterCenter(found_cluster_key)
            speed = (new_cluster_center - old_cluster_center) * 1 / self.delta_t
            #print("End of function",len(self.active_points))
            return [True, new_cluster_center, speed]
        else:
            #print("End of function",len(self.active_points))
            return [False]
    def getClusterCenter(self, cluster_key):
        num_points = 0
        summe = np.array([0,0,0])
        for d_point in self.active_clusters[cluster_key]:
            num_points += 1
            summe = summe +np.array(d_point[0])
        center = summe / float(num_points)
        return center
    
    def fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img
    
    def getAllClusters(self):
        return self.all_clusters
    
    def getDataPoints(self):
        return self.all_data_points
    
    def get_Plotframes(self):
        return self.plots
        
    def getNeighbors(self, P):
        N=[]
        for point in self.active_points:
            a = np.array((P[0:3]))
            b = np.array((point[0:3]))
            dist = np.sqrt(np.sum(np.square(a-b)))
            
            if float(dist) <= float(self.max_dist):
                N.append(point)
        return N
    