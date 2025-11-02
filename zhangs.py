import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


output_dir = "./zhangsOutput"
param_results= "./zhangsOutput/paramResults.txt"
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
        print(f"Skipping {img_path}, corners not found")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2)

H_list=[]
#compose H by using DLT
def compute_H(objpoints,imgpoints):
    global H_list
    for i in range(len(objpoints)):
        Q_list=[]
        obj_pts = objpoints[i]  # objpoints is a list with numpy inside
        img_pts = imgpoints[i]
        for pt_idx in range(len(obj_pts)):
            X = obj_pts[pt_idx, 0]
            Y = obj_pts[pt_idx, 1]
            u = img_pts[pt_idx, 0]
            v = img_pts[pt_idx, 1]

            # Q 2행 append
            Q_list.append([X, Y, 1, 0, 0, 0, -u*X, -u*Y, -u])
            Q_list.append([0, 0, 0, X, Y, 1, -v*X, -v*Y, -v])
        U, S, Vt = np.linalg.svd(Q_list)
        h = Vt[-1, :]           
        H = h.reshape(3,3)
        H /= H[2,2]             
        H_list.append(H)
    return H_list

def v_ij(h_i, h_j):
    return np.array([h_i[0]*h_j[0], h_i[0]*h_j[1]+h_i[1]*h_j[0], 
            h_i[1]*h_j[1], h_i[2]*h_j[0]+h_i[0]*h_j[2],
            h_i[2]*h_j[1]+h_i[1]*h_j[2], h_i[2]*h_j[2]])
    

def compute_K(H_list):
    V=[]
    for H in H_list:
        h1 = H[:,0]  
        h2 = H[:,1]

        V.append(v_ij(h1,h2))
        V.append(v_ij(h1,h1)-v_ij(h2,h2))
    U, S, Vt = np.linalg.svd(V)
    b=Vt[-1,:]
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])
    K=np.linalg.inv(np.linalg.cholesky(B).T)
    #normalize K so that last element is 1
    K = K / K[2, 2]  
    return K

R_list,T_list=[],[]
def compute_ex(K,H_list):
    global R_list, T_list
    K_inv=np.linalg.inv(K)
    for H in H_list:
        h1 = H[:,0]  
        h2 = H[:,1]
        h3 = H[:,2]

        lam = 1.0 / np.linalg.norm(np.dot(K_inv, h1))

        r1=lam * np.dot(K_inv, h1)
        r2=lam * np.dot(K_inv, h2)
        t=lam * np.dot(K_inv, h3)
        r3=np.cross(r1,r2)
        R=np.column_stack([r1,r2,r3])
        #make R orthogonal by SVD (adjusting noise)
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R[:,2] *= -1

        R_list.append(R)
        T_list.append(t)

        #check if R is orthogonal
        #print("this should be true:", np.allclose(R.T @ R, np.eye(3), atol=1e-6))
        #print("this shld be 1:", np.linalg.det(R))
        #print("this is t", t)

    return R_list,T_list

P_list=[]
def reprojection_error(objpoints,imgpoints,K,R_list,T_list):
    global P_list
    total_error=0
    for i in range(len(objpoints)):
        R=R_list[i]
        t=T_list[i]
        # P is projection matrix (3D > 2D)
        P=K @ np.hstack([R,t.reshape(3,1)])
        img_proj=[]
        for j,each in enumerate(objpoints[i]):
            hom=np.hstack([each,1]).reshape(4,1)
            proj=np.dot(P,hom)

            #normalization
            proj /= proj[2]
            img_proj.append(proj[:2].ravel())
            total_error += np.linalg.norm(proj[:2].ravel() - imgpoints[i][j])
        P_list.append(np.array(img_proj))
    #avg error
    mean_error=total_error / sum(len(pts) for pts in objpoints)
    return P_list,mean_error


    
#create imgpoint/objpoint array for all data imgs
for img_path in data:
    img=cv2.imread(img_path)
    corners=detect_corners(img,pattern_size)
    objpoints.append(objp)
    imgpoints.append(corners)

H_list=compute_H(objpoints,imgpoints)
K=compute_K(H_list)
R_list,T_list=compute_ex(K,H_list)
P_list,error=reprojection_error(objpoints,imgpoints,K,R_list,T_list)

#print parameters
with open(param_results,"w") as f:
    f.write(f"Mean reprojection error: {round(error,4)}\n\n")
    f.write(f"Intrinsic Matrix K: {K}\n")
    for i , img_path in enumerate(data):
        img_name = os.path.basename(img_path)
        f.write(f"for {img_name} \n") 
        f.write(f"Rotation Matrix R: {R_list[i]}\n") 
        f.write(f"Translation vector t: {T_list[i].ravel()}\n\n") 

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






