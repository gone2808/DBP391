#Library
import numpy as np
import cv2 as cv
import math
import argparse
from TrajectoryExtraction import TrajectoryExtraction 
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans,DBSCAN
# Method 
DBSCAN_KMEANS=0
AREA_THRESHOLD=0.15
FLOW_DIRECTION_THRESHOLD=1.1
POSITION_THRESHOLD=1.4
# Plot
PLOT_GOOD_TRAJECTORY=0
PLOT_CLUSTERED=0
def plotTraject(video_folder_path, video_name, trajects, traject_alias, colorMap=None):
    """
        plot traject on first frame
    """
    cap = cv.VideoCapture(video_folder_path + video_name)
    _,frame = cap.read()
    
    if (colorMap is not None):
        color = np.random.randint(0, 255, (max(colorMap)+1, 3))
    else:
        color = np.random.randint(0, 255, (len(trajects), 3))
    for i in range(len(trajects)):
        colorIndex = i
        if (colorMap is not None):
            colorIndex = colorMap[i]
        if(colorIndex==-1): continue
        (x0,y0) = trajects[i][0]
        frame = cv.circle(frame, (int(x0), int(y0)), 3, color[colorIndex].tolist(), -1)
        for j in range(len(trajects[i])-1):
            (x,y) = trajects[i][j+1]
            (px,py) = trajects[i][j]
            frame = cv.line(frame, (int(px), int(py)), (int(x), int(y)),color[colorIndex].tolist(), 2)

    cv.imshow(traject_alias, frame)

    while(1):
        if cv.waitKey(1) == 27:
            break
def getSegment(video_folder_path, video_name, trajects, colorMap=None):
    """
    return binary image of Trajectory Cluster Segment
    """
    cap = cv.VideoCapture(video_folder_path + video_name)
    _,frame = cap.read()
    mask=np.zeros_like(frame)
    for i in range(len(trajects)):
        colorIndex = i
        if (colorMap is not None):
            colorIndex = colorMap[i]
        if(colorIndex==-1): continue
        for j in range(len(trajects[i])-1):
            (x,y) = trajects[i][j+1]
            (px,py) = trajects[i][j]
            mask = cv.line(mask, (int(px), int(py)), (int(x), int(y)),255, 3)
    
    kernel = np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]],np.uint8)
    mask = cv.dilate(mask,kernel,iterations = 1)
    kernel = np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]],np.uint8)
    mask=cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    (thresh, mask) = cv.threshold(mask, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # cv.imshow(traject_alias, mask)
    # while(1):
    #     if cv.waitKey(1) == 27:
    #         break

    return mask


def calcPathLength(pathArray):
    apts = np.array(pathArray) # Make it a numpy array
    lengths = np.sqrt(np.sum(np.diff(apts, axis=0)**2, axis=1)) # Length between corners
    return np.sum(lengths)

def angle_trunc(a):
    while a < 0.0:
        a += math.pi * 2
    return a

def getAngleBetweenPoints(x_orig, y_orig, x_landmark, y_landmark):
    deltaY = y_landmark - y_orig
    deltaX = x_landmark - x_orig
    return angle_trunc(math.atan2(deltaY, deltaX))

def euclidDistance(a,b,c,d):
    return math.sqrt((c-a)**2+ (d-b)**2)

def trajectFilter(trajects):
    MIN_TRAJECT_DISTANCE = 30
    MIN_EUCLID_DISTANCE = 10
    return [a for a in trajects if (calcPathLength(a) >= MIN_TRAJECT_DISTANCE) and (euclidDistance(a[0][0],a[0][1],a[-1][0],a[-1][1])>MIN_EUCLID_DISTANCE)]

def expand(m,i,ni,visited,featureSet,segment,member,final):
    member[i]=m
    try:
        final[m]=cv.bitwise_or(final[m],segment[i])
    except:
        print(segment[i])
    for j in ni:
        if not visited[j]:
            for z in range(len(visited)):
                if z!=j and visited[z] and member[z]!=-1:
                    continue
                # AREA
                overlap_area=cv.countNonZero(cv.bitwise_and(segment[z],segment[j])) / min(cv.countNonZero(segment[z]),cv.countNonZero(segment[j]))
                #POS
                # POSITION
                xz=np.mean([_[8] for _ in featureSet[z]])
                yz=np.mean([_[9] for _ in featureSet[z]])
                xj=np.mean([_[8] for _ in featureSet[j]])
                yj=np.mean([_[9] for _ in featureSet[j]])
                position_distance=euclidDistance(xz,yz,xj,yj)
                #FLOW
                sj=np.mean([_[10] for _ in featureSet[j]])
                cj=np.mean([_[11] for _ in featureSet[j]])
                sz=np.mean([_[10] for _ in featureSet[z]])
                cz=np.mean([_[11] for _ in featureSet[z]])
                flow_direction_distance=euclidDistance(sj,cj,sz,cz)
                if overlap_area >=AREA_THRESHOLD and position_distance<=POSITION_THRESHOLD and flow_direction_distance <= FLOW_DIRECTION_THRESHOLD:
                    ni.append(z)
            visited[j]=True
        if member[j]==-1:
            member[j]=m
            final[m]=cv.bitwise_or(final[m],segment[j])
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')

    """
    DATA
    """
    video_folder_path = "Data/"
    
    #check  0 15% 3  1 15% 1.8
    # video_name = "701-111_l.mov" 
    #check  0 15% 3    1 15% 1.8
    # video_name = "2.webm"
    # check 0 15% 3    1 15% 1.8
    # video_name = "4.webm"
    #  check  0 15% 3 1 15% 1.8
    # video_name = "001-0438.mov"
    # check 0 15%  3   1 15% 1.8
    # video_name = "3687-18_70.mov"
    # check   0 15% 1.4/2    1 15% 1.4
    # video_name = "rush_01.mov"
    # video_name="crazy.mov"
    # video_name = "4917-5_70.mov"
    # video_name = "620-72_l.mov"
    # video_name = "662-5_l.mov"
    video_name="2.webm"
    parser.add_argument('-p',
                        '--video_folder_path',
                        type=str,
                        required=False,
                        default='Data/',
                        help='Video Folder Path')
    parser.add_argument('-n',
                        '--video_name',
                        type=str,
                        required=False,
                        default='5.webm',
                        help='Video Name')
    parser.add_argument('-f',
                        '--frame_limit',
                        type=int,
                        required=False,
                        default=-1,
                        help='Limit extraction frame')
    parser.add_argument('-m',
                        '--method',
                        type=int,
                        required=False,
                        default=0,
                        help='method')
    parser.add_argument('-pf',
                        '--plot_filtered',
                        type=int,
                        required=False,
                        default=0,
                        help='plot_filtered')
    parser.add_argument('-pc',
                        '--plot_clustered',
                        type=int,
                        required=False,
                        default=0,
                        help='plot_clustered')
    parser.add_argument('-at',
                        '--area_threshold',
                        type=float,
                        required=False,
                        default=0.15,
                        help='area_threshold')
    parser.add_argument('-ft',
                        '--flow_threshold',
                        type=float,
                        required=False,
                        default=1.1,
                        help='flow_threshold')
    parser.add_argument('-pt',
                        '--position_threshold',
                        type=float,
                        required=False,
                        default=1.4,
                        help='position_threshold')
    args=parser.parse_args()
    video_folder_path = args.video_folder_path
    video_name=args.video_name
    DBSCAN_KMEANS=args.method
    PLOT_GOOD_TRAJECTORY=args.plot_filtered
    PLOT_CLUSTERED=args.plot_clustered
    AREA_THRESHOLD=args.area_threshold
    FLOW_DIRECTION_THRESHOLD=args.flow_threshold
    POSITION_THRESHOLD=args.position_threshold
    """
    =====================
    TRAJECTORY EXTRACTION
    =====================
    """
    te = TrajectoryExtraction(video_folder_path, video_name,)
    te.getTrajects(stopFrameNum=args.frame_limit)
    """
    ====================
    TRAJECTORY FILTERING
    ====================
    """
    goodTrajects = trajectFilter(te.trajects)
    if PLOT_GOOD_TRAJECTORY:
        plotTraject(video_folder_path, video_name, goodTrajects, "Filtered Trajectory")
    """
    =======================
    CALC TRAJECTORY FEATURE
    =======================
    """
    features = []
    for traject in goodTrajects:
        x = [i[0] for i in traject]
        y = [i[1] for i in traject]
        t = range(len(traject))
        # Polynomial 
        a = np.polyfit(x, t, 3)
        b = np.polyfit(y, t, 3)
        # Position
        position = [np.mean(x), np.mean(y)]
        # Flow Direction 
        #Default
        # direction= [min(max(y[-1]-y[0],-1),1)] 
        # Complex Number
        angle=getAngleBetweenPoints(x[0],y[0],x[-1],y[-1])
        direction= [ math.cos(angle),math.sin(angle)]  


        # [ Polynomial , Position , Flow Direction ]
        features.append(np.concatenate([a,b,position,direction]))

    #Features Count
    featuresCount=len(features)
    """
    ==========================
    GROUPING FOR CRUDE CLUSTER
    ==========================
    DBSCAN_KMEANS

    add dbscan_parameter then grouped by kmeans 
    """

    # Parameter Scaling 
    features=scale(features,axis=0)
    if DBSCAN_KMEANS:
        #Calc Euclid Distance for eps choosing ( Median )
        featureDistances=[]
        for i in range(featuresCount-1):
            for j in range(i+1,featuresCount):
                featureDistances.append(np.linalg.norm(features[i] - features[j]))
        # Sort for median distance
        sortedDistance=featureDistances
        sortedDistance.sort()
        # Pick 3 median n/4 n/8 n/16
        eps=[sortedDistance[int(len(sortedDistance)/16)],sortedDistance[int(len(sortedDistance)/8)],sortedDistance[int(len(sortedDistance)/4)]]
        # eps=[0.6,0.3,0.15]
        # Density ( DBScan)
        features2=None
        for e in eps:
            dbscan=DBSCAN(eps=e).fit(features)
            if features2 is None:
                features2=np.reshape(dbscan.labels_,(-1,1))
            else:
                features2=np.concatenate((features2,np.reshape(dbscan.labels_,(-1,1))),axis=1)   
    
        # [ Polynomial , Position , Flow Direction , Density ]
        totalFeatures=np.concatenate((features, features2), axis=1)
    else:
        totalFeatures=features
    # Scaling for Clustering
    totalFeatures=scale(totalFeatures,axis=0)
    if DBSCAN_KMEANS:
        clusterResult = KMeans(n_clusters=int(featuresCount/15), random_state=0).fit(np.array(totalFeatures))
    else:
        # featureDistances=[]
        # for i in range(featuresCount-1):
        #     for j in range(i+1,featuresCount):
        #         featureDistances.append(np.linalg.norm(totalFeatures[i] - totalFeatures[j]))
        # # DBSCAN
        # sortedDistance=featureDistances
        # sortedDistance.sort()
        # # Pick eps
        # eps= np.mean(sortedDistance)/7  #sortedDistance[int(len(sortedDistance)*0.02)]
        # np.savetxt("eps.txt",np.array([eps]))
        # 0.55
        clusterResult = DBSCAN(eps=0.35,min_samples=3).fit(totalFeatures)
    
    clusterLabel = clusterResult.labels_
    # print(clusterLabel)
    """
    ===========
    PLOT RESULT
    ===========`
    """
    if PLOT_CLUSTERED:
        plotTraject(video_folder_path, video_name, goodTrajects, "Crude Cluster", colorMap=clusterLabel)

    clusterSet = []
    featureSet = []
    for i in range(max(clusterLabel)):
        cluster = [goodTrajects[j] for j in range(len(goodTrajects)) if clusterLabel[j]==i]
        feature = [features[j] for j in range(len(goodTrajects)) if clusterLabel[j]==i]
        clusterSet.append(cluster)
        featureSet.append(feature)

    clusterSet.sort(key=lambda s: len(s), reverse=True)
    featureSet.sort(key=lambda s: len(s), reverse=True)
    """
    ==================
    CALC CRUDE SEGMENT
    ==================
    """
   
    if DBSCAN_KMEANS:
         clusterSet_scale=0.5
    else:
        clusterSet_scale=0.95
    segment=[]
    for i in range(int(len(clusterSet)*clusterSet_scale)):
        segment.append(getSegment(video_folder_path, video_name, clusterSet[i]))
    # for i in range(int(len(clusterSet))):
    #     plotTraject(video_folder_path, video_name, clusterSet[i], "trajects of a dominant cluster")

    """
    ==================
    FINAL CLUSTERING
    ==================
    """
    cap = cv.VideoCapture(video_folder_path + video_name)
    _,frame = cap.read()
    mask=np.zeros_like(frame)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    visited=[False]*len(segment)
    final=[mask]*len(segment)
    member=[-1]*len(segment)
    m=0
    for i in range(int(len(segment))):
        if visited[i]: continue
        ni=[]
        for j in range(int(len(segment))):
            if(i!=j):
                # AREA
                overlap_area=cv.countNonZero(cv.bitwise_and(segment[i],segment[j])) / min(cv.countNonZero(segment[i]),cv.countNonZero(segment[j]))
                # POSITION
                xi=np.mean([_[8] for _ in featureSet[i]])
                yi=np.mean([_[9] for _ in featureSet[i]])
                xj=np.mean([_[8] for _ in featureSet[j]])
                yj=np.mean([_[9] for _ in featureSet[j]])
                position_distance=euclidDistance(xi,yi,xj,yj)
                # DIRECTION
                si=np.mean([_[10] for _ in featureSet[i]])
                ci=np.mean([_[11] for _ in featureSet[i]])
                sj=np.mean([_[10] for _ in featureSet[j]])
                cj=np.mean([_[11] for _ in featureSet[j]])
                flow_direction_distance=euclidDistance(si,ci,sj,cj)
                if  overlap_area>=AREA_THRESHOLD and position_distance<= POSITION_THRESHOLD and flow_direction_distance<= FLOW_DIRECTION_THRESHOLD:
                    ni.append(j)
        expand(m,i,ni,visited,featureSet,segment,member,final)
        visited[i]=True
        m=m+1
    finalSegment=np.zeros_like(frame)
    color = np.random.randint(0, 255, (len(segment), 3))
    # color = np.array([(60, 76, 231),(34, 126, 230),(15, 196, 241),(156, 188, 26),(113, 204, 46),(219, 152, 52),(182, 89, 155),(18, 156, 243),(0, 84, 211), (43, 57, 192),(133, 160, 22),(96, 174, 39),(185, 128, 41),(173, 68, 142)])
    finalSegment[:]=(69,11,11)
    for i in range(m):
        tmp=np.zeros_like(frame)
        tmp[final[i]>0]=color[i]
        finalSegment=finalSegment+tmp
    cv.imshow("Segmentation",finalSegment)
    while(1):
        if cv.waitKey(1) == 27:
            break
    cv.destroyAllWindows()
