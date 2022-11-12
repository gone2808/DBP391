import math
import numpy as np
import cv2 as cv

# params for Shi-Tomasi corner detection
feature_params = dict( maxCorners = 10000,   
                            qualityLevel = 0.05, 
                            minDistance = 1,     
                            blockSize =5)       
        
# Parameters for lucas kanade optical flow
lk_params = dict( winSize =(15, 15),
                maxLevel = 2,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                            10, 0.03))


# def getAngle2(a, b, c):
#     ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
#     return abs(180-ang)
    

def getAngle(a, b, c):
		"""
				Returns the angle between the vectors ab and bc
		"""
		ba = a - b
		bc = c - b
		cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
		angle = np.arccos(cosine_angle)
		print(angle)
		return angle


class TrajectoryExtraction:
    # Corner Detection Shi-Tomasi
    # Tracking Lucas Kanade
    def __init__(self, vid_dir, vid_name):
        self.VID_DIR=vid_dir
        self.VID=vid_name
        self.frameCount=0  
        # Update Point
        self.FLAG_RADIUS = 6 # mark detected point to avoid duplicate
        self.TIME_UPDATE = 1.5 # time to update new point
        self.DETECT_THRESHOLD = 0.015
        #Plot
        self.MIN_TRAJECTORY_DISTANCE=0 # very-short traject will not be plotted
        #Track
        self.OUT_THRESHOLD=0.01
        self.prev_gray=None
        self.colorId=[]
        self.tracks=[] # temporary tracking corners
        self.trajects=[] # completed trajectories (output of class)

    def getTrajects(self, stopFrameNum=-1):
        cap = cv.VideoCapture(self.VID_DIR + self.VID)

        input_fps = cap.get(cv.CAP_PROP_FPS)
        frame_refresh=int(input_fps*self.TIME_UPDATE) # update new corner every N frame should increase depend on cam fps
        frameId = -1

        traj_frame = None
        # Create some random colors
        color = np.random.randint(0, 255, (10000, 3))   
        maxId=0
        while(1):
            #Read Frame
            _,frame = cap.read()
            if frame is None:
                break
            frameId += 1
            
            h,w,c=frame.shape
            # frame=frame[:,10:710]
            gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            if len(self.tracks)>0:
                # Create a mask image for drawing purposes
                if traj_frame is None:
                    traj_frame = np.zeros_like(frame)
                # calculate optical flow  ( add back tracking)
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st,_err = cv.calcOpticalFlowPyrLK(self.prev_gray,gray,p0, None,**lk_params)
                p0r,_st,_err = cv.calcOpticalFlowPyrLK(gray,self.prev_gray,p1, None,**lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                keep=[True]*len(self.tracks)
                for i,( (x, y), good_flag) in enumerate ( zip( p1.reshape(-1, 2), good ) ):
                    if not good_flag:
                        continue
                    stmp=self.tracks[i][0]
                    ptmp=self.tracks[i][-1]
                    sx,sy=stmp[0],stmp[1]
                    px,py=ptmp[0],ptmp[1]
                    if ((h*self.OUT_THRESHOLD >= y) or (h*(1-self.OUT_THRESHOLD)<=y) \
                            or (w*self.OUT_THRESHOLD >= x) or (w*(1-self.OUT_THRESHOLD)<=x)):
                        keep[i]=False
                        continue
                    # REMOVE ANOMALY
                    if len(self.tracks[i]) >1:
                        pptmp=self.tracks[i][-2]
                        ppx,ppy=pptmp[0],pptmp[1]
                        if  math.sqrt((x-px)**2+(y-py)**2) >30 and getAngle((ppx,ppy),(px,py),(x,y))>30:
                            keep[i]=False
                            continue
                    #track 
                    # if max(abs(x-px),abs(y-py))>2:
                    #     self.tracks[i].append((x, y))
                    self.tracks[i].append((x, y))
                    if ( math.sqrt((x-sx)**2+(y-sy)**2) >= self.MIN_TRAJECTORY_DISTANCE):
                        try:
                            traj_frame = cv.line(traj_frame, (int(px), int(py)), (int(x), int(y)),color[self.colorId[i]].tolist(), 2)
                        except Exception as e:
                            print(e)
                #Update new tracks
                newTracks=[]
                newId=[]
                for i,item in enumerate(self.tracks):
                    if keep[i]:
                        newTracks.append(item)
                        newId.append(self.colorId[i])
                    else:
                        self.trajects.append(item)
                self.tracks=newTracks
                self.colorId=newId
            if(self.frameCount%frame_refresh==0):
                mask=np.zeros_like(gray)
                mask[:] = 255
                mask[int(h*self.DETECT_THRESHOLD):int(h*(1-self.DETECT_THRESHOLD)),int(w*self.DETECT_THRESHOLD):int(w*(1-self.DETECT_THRESHOLD))]  = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), self.FLAG_RADIUS, 0, -1)
                p = cv.goodFeaturesToTrack(gray, mask = mask,**feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                        self.colorId.append(maxId)
                        maxId+=1
            
            #prepare for next
            self.prev_gray = gray
            self.frameCount+=1
            if traj_frame is not None:
                img = cv.add(frame, traj_frame)
                cv.imshow('Extracting Trajectory', img)

            if cv.waitKey(1) == 27 or frameId == stopFrameNum:
                break
        
        for item in self.tracks:
            self.trajects.append(item)

if __name__ == "__main__":
    video_folder_path = "Data/"
    video_name = "4917-5_70.mov"

    te = TrajectoryExtraction(video_folder_path, video_name)
    te.getTrajects()
    
    while(1):
        if cv.waitKey(1) == 27:
            break
    cv.destroyAllWindows()