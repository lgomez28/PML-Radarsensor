from DataGenerationRadar3D import RadarSensor, Target
import numpy as np
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
from KalmanFilter import KalmanFilter
from PIL import Image
import cv2 as cv2
import random as rand
from DBScan import DBSCAN
from DBScanner import DBScanner
import time

# Definition Globale Variablen
color_list=['c','m','r','g','b','y','darkred', 'darkolivegreen', 'gold','lime','slategray']
Plotframes = []
MinPts=3
eps = 0.4

#Definition Funktionen
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
def plot_datapoints3d(points_list):
    points_array = np.array(points_list)
    x_points = points_array[:,0]
    y_points = points_array[:,1]
    z_points = points_array[:,2]
    fig=pyplot.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(0,0,0,s=1,marker='s',color='y')
    ax.scatter(x_points, y_points, z_points,s=0.7,color='r')

def plot_Clusterdict3d(clusterdict, save_Plot = False):
    print("PLOT ")
    fig=pyplot.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(0,0,0,s=1,marker='s',color='y')    
    for key in clusterdict.keys():
        points_list = list(clusterdict[key])
        points_array = np.array(points_list)
        print(points_array.shape)
        if len(points_array.shape) >= 3:
            points_array = points_array[:,0,:]
        x_points = points_array[:,0]
        y_points = points_array[:,1]
        z_points = points_array[:,2]
        ax.scatter(x_points, y_points, z_points,s=0.7,color=color_list[key])
    if save_Plot:
        Plotframes.append(fig2img(fig))

def get_random_visible_point():
    y_init = rand.uniform(0.,12.)
    x_init = rand.uniform(-y_init*0.15,y_init*0.15)
    z_init = rand.uniform(-y_init*0.15,y_init*0.15)
    initalposition = [x_init,y_init,z_init]
    return initalposition
def get_dist(x,y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)

def make_tagets(num_targets, num_points_per_target, max_point_distance,num_crossing_paths = 6, close_starts = False):
    targets = []
    initialpositions = []
    paths = []
    velss = []
    for i in range(0, num_targets):
        initialposition = np.array(get_random_visible_point())
        all_distances_cehck = False
        while not all_distances_cehck:
            initialposition = np.array(get_random_visible_point())
            all_distances_cehck = True
            for initialp in initialpositions:
                dist = get_dist(initialp, initialposition)
                if dist <= 0.5:
                    all_distances_cehck = False
            
        initialpositions.append(initialposition)
        path = []
        tempos = initialposition
        vels = np.ones((1,num_points_per_target))
        while len(path) < num_points_per_target:
            new_pos = get_random_visible_point()
            dist = np.sqrt((tempos[0] - new_pos[0]) ** 2 + (tempos[1] - new_pos[1]) ** 2 + (tempos[2] - new_pos[2]) ** 2)
            if dist < max_point_distance:
                path.append(new_pos)
                tempos = new_pos
                vel = rand.uniform(0.5, 1.3)
                vels[0,len(path)-1]=vel
        velss.append(vels)
        #print(initialposition,path,vels)
        paths.append(path)
    for i in range(0, num_crossing_paths):
        paths[rand.randint(0, len(paths)-1)] [rand.randint(0, len(paths)-1)] = paths[rand.randint(0, len(paths)-1)] [rand.randint(0, len(paths)-1)]
    for i in range(0, num_targets):
        opt = {
                'InitialPosition' : initialpositions[i],
                'Path' : np.array(paths[i]).transpose(),
                'Velocities' : velss[i]
            }
        target = Target(opt)
        targets.append(target)
    return targets


def random_recursive_optimation3D(best_error, qerr, rk,pred_err_multiplier, recursive_depth, filter_input, trys_per_recursion,ground_truth_dict):
    counter = 0
    while counter < trys_per_recursion:
        
        qerr_rand = qerr + np.array([[rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth)),
                                rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth)),
                                rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth))],
                                [rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth)),
                                rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth)),
                                rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth))],
                                [rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth)),
                                rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth)),
                                rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth))]])
        rkd1 = rk[0,0] + rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth))
        rkd2 = rk[1,1] + rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth))
        rkd3 = rk[2,2] + rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth))
        rkv1 = rk[0,1] + rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth))
        rkv2 = rk[0,2] + rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth))
        rkv3 = rk[1,2] + rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth))
        rk_rand = np.array([[rkd1,rkv1,rkv2], [rkv1,rkd2, rkv3], [rkv2, rkv3, rkd3]])
        pred_err_multiplier_rand =  pred_err_multiplier + rand.uniform(-10000 / pow(2,recursive_depth), 10000 / pow(2,recursive_depth))
        errorsum = 0
        error_mean = 0
        temp_results = {}
        for cluster in filter_input.keys():
            errorsum = 0
            temppos = [0,0,0]
            kFilter = KalmanFilter(qerr_rand, rk_rand, pred_err_multiplier_rand)
            if recursive_depth > 5:
                kFilter = KalmanFilter(qerr_rand, rk_rand, pred_err_multiplier_rand, True)
            temp_results[cluster] = []
            i = 0
            if i > len(ground_truth_dict[cluster]):
                turth = ground_truth_dict[cluster][0]
            for clustercenter, speed in filter_input[cluster]:
                #print([clustercenter[0],clustercenter[1],clustercenter[2]], temppos,speed)
                dist,vel = kFilter.Step([clustercenter[0],clustercenter[1],clustercenter[2]], temppos,speed)
                temp_results[cluster].append(dist)
                temppos=[clustercenter[0],clustercenter[1],clustercenter[2]]
                lowest_dist = 99999
                if i < len(ground_truth_dict[cluster]):
                    turth = ground_truth_dict[cluster][i]
                temp_dist = get_dist(dist, turth)
                if temp_dist < lowest_dist:
                    lowest_dist = temp_dist
                errorsum += lowest_dist
                i += 1
            error_mean += errorsum*errorsum / float(i)
        if error_mean < best_error:
            increase = best_error / error_mean
            best_error = error_mean
            qerr = qerr_rand
            rk = rk_rand
            pred_err_multiplier = pred_err_multiplier_rand
            if increase > 1.01:
                print("\n\n\nRandom for the Win :D rk:",rk,"\n qerr:",qerr, "\n\n Neuer Bester Error : ", best_error, "\n\n Neuer Bester Pred_err_multiplier : ", pred_err_multiplier)
                plot_Clusterdict3d(temp_results, True)
        counter += 1
        if counter / float(trys_per_recursion) in [0.1, 0.3, 0.4, 0.5, 0.7, 0.9]:
            print("Recursion Progress:", counter / float(trys_per_recursion))
    recursive_depth += 1
    if recursive_depth >= 15:
        return qerr, rk,best_error
    return random_recursive_optimation3D(best_error, qerr, rk,pred_err_multiplier, recursive_depth, filter_input, trys_per_recursion,ground_truth_dict)


# Dictionary mit Ground Truths
gt_pos_dict={}

# Erueugen von Targets

targets = make_tagets(3,8,1.3, 0, False)#[x, x2]#,x3]
truth = {}
i = 0
for target in targets:
    truth [i] = target
    
    i += 1

# Erzeugen von Listen für Ground Truths für jedes Target

for i in range(0,len(targets)):
    gt_pos_dict[i]=[]


'''
Setup the radar sensor
The radar sensor points always to the direction along the y axis
(see diagram in the note)
'''

optRadar = {
    'Position' : np.array([0,0,0]),                    # numpy.array([0,0,0.5]) 
    'OpeningAngle' : np.array([120,90]), # [Horizontal, Vertical]
    'FalseDetection': True
}
sensor = RadarSensor(optRadar)
getNext = True
dets = []

# Initialisierung des DB Scanners
scanner = DBScanner(MinPts, eps, num_active_points = 150, num_active_points_per_cluster = 3)

# Iterieren über die vom Scanner whrgenommenen Punkte.
all_detections = []
while(getNext == True):
    targetCounter = 0
    for target in targets:
        currentpos, currentvel = target.Step(1/sensor.opt['MeasurementRate'])
        getNext = getNext & ~target.reachedEnd    
############################################################## Groundtruth   
        gt_pos_dict[targetCounter].append([currentpos[0], currentpos[1] , currentpos[2]-optRadar['Position'][2]])
        targetCounter += 1 
############################################################## detected values  [x,y,z,vel]    
    dets = sensor.Detect(targets)
    # Ansatz mit bewegten Clusterzentren zur GEschwindigkeitsberrechnung
    dets_on_speed = []
    for x in dets:
        all_detections.append(x)
        scanInfo = scanner.add_data_point(x)
        if scanInfo[0]:
            clustercenter= scanInfo[1]
            speed = scanInfo[2]
            dets_on_speed.append([clustercenter, speed])
            
# dets = ALLE PUNKTE
# dets_on_speed = GECLUSTERETE PUNKTE MIT GESCHWINDIGKEIT
# clusterdict = DICTIONARY MIT PUNKT UND VEL PRO CLUSTER
cluster_dict = scanner.getAllClusters()


clustered_data_points = scanner.clustered_data_points
plot_datapoints3d(clustered_data_points)


# initial Identiys als startpunkt der rekursiven oOtimierung
qerr = np.identity(3)
rk = np.array([[1,1,1],[4,5,6],[7,8,9]])

clustered_data_points = scanner.clustered_data_points
plot_datapoints3d(clustered_data_points)
# Plotten der Daten vor Kalman Filterung
plot_Clusterdict3d(gt_pos_dict)
plot_datapoints3d(all_detections)
plot_Clusterdict3d(cluster_dict)
#def random_recursive_optimation3D(best_error, qerr, rk, recursive_depth, filter_input, trys_per_recursion,ground_truth_dict):
random_recursive_optimation3D(9999999999, qerr, rk, 500,  0, cluster_dict, 10000, gt_pos_dict)

        

videodata = Plotframes
video_name = "C:/Users/jgaertner/Downloads/10_12/RadarSensor3D (1)/RadarSensor3D/Video_test_Kalmann.avi"
height, width, layers = np.array(videodata[0]).shape
video = cv2.VideoWriter(video_name, 0, 40, (width,height))
for frame in videodata:
    im_array = np.array(frame)[...,:3]
    print(im_array.shape)
    video.write(im_array)
print(type(videodata[0]))
#cv2.destroyAllWindows()
video.release()



