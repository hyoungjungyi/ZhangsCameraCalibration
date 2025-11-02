import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


output_dir = "./openCVOutput"
param_results= "./openCVOutput/paramResults.txt"
os.makedirs(output_dir, exist_ok=True)
data = glob.glob("./data/*.jpg")

pattern_size = (13, 9)
square_size = 2.0

#world coord
objp = np.zeros((pattern_size[0]*pattern_size[1],3),np.float32)
objp[:,:2]=np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
objp *= square_size

#3d points, 2d points
objpoints=[]
imgpoints=[]


def detect_corners(img,pattern_size=(13,9)):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,corners=cv2.findChessboardCorners(gray, pattern_size,None)
    if not ret:
        print(f"Skipping, corners not found")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2)

H_list=[]
def reprojection_error(objpoints,imgpoints,K,rvecs,tvecs,dist):
    P_list=[]
    total_error=0
    for i in range(len(objpoints)):
        #instead of simple projection, adjust distortion by using cv2.projectPoints
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1,2)
        P_list.append(proj)
        for j in range(len(objpoints[i])):
            total_error += np.linalg.norm(proj[j] - imgpoints[i][j])
        
    #avg error
    mean_error=total_error / sum(len(pts) for pts in objpoints)
    return P_list,mean_error
    
#create imgpoint/objpoint array for all data imgs
for img_path in data:
    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners=detect_corners(img,pattern_size)
    objpoints.append(objp)
    imgpoints.append(corners)
    #camera calibration
ret, K , dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
P_list,error=reprojection_error(objpoints,imgpoints,K,rvecs,tvecs,dist)

#print parameters
with open(param_results,"w") as f:
    f.write(f"Mean reprojection error: {round(error,4)}\n\n")
    f.write(f"Intrinsic Matrix K: {K}\n")
    f.write(f"Distortion coefficients: {dist}\n")
    for i , img_path in enumerate(data):
        img_name = os.path.basename(img_path)
        R,_=cv2.Rodrigues(rvecs[i])
        f.write(f"for {img_name} \n") 
        f.write(f"Rotation Matrix R: {R}\n") 
        f.write(f"Translation vector t: {tvecs[i].ravel()}\n\n") 

#result visualization
for i, img_path in enumerate(data):
    img_pts = imgpoints[i]
    reproj_pts = P_list[i]

    img=cv2.imread(img_path)
    for pt in reproj_pts:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), radius=7, color=(0,0,255), thickness=5)
    for pt in img_pts:
        u, v = int(pt[0]), int(pt[1])
        cv2.circle(img, (u,v), radius=7, color=(255,0,0), thickness=5)
    fname = os.path.basename(img_path)
    out_path = os.path.join(output_dir, f"reprojection_{fname}")
    cv2.imwrite(out_path, img)

    plt.scatter(img_pts[:,0], img_pts[:,1], c='r', s=20)
    plt.scatter(reproj_pts[:,0], reproj_pts[:,1], c='b',marker='x', s=20, label='reprojected points', zorder=20)
plt.legend(["detected corners", "reprojected points"])
plt.savefig(os.path.join(output_dir, "reprojection_result.png"), dpi=300)  
plt.close()






