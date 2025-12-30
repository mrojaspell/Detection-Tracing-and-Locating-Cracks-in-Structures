# imports
import numpy as np
import scipy.ndimage
import skimage
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy 


##############################################
### Provided code - nothing to change here ###
##############################################

from pylab import *
from scipy import signal
from scipy import *
import numpy as np
from PIL import Image

pic1 = 'crack_images/crack6.jpg'
pic2 = 'small_crack/crack6_small.jpg'

# Load both images, convert to double and grayscale
def getSIFTKP(file, n):
    img = cv.imread(file)
    #img = img*4 - 2*scipy.ndimage.gaussian_filter(img,sigma=1)
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
 
    sift = cv.SIFT_create()
    k,d = sift.detectAndCompute(gray, None)
    output_image = cv.drawKeypoints(gray, k, 0, (255, 0, 0), 
                                 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
    
    plt.imshow(output_image)
    plt.show()

    return k,d,img

def getSIFTKPfromIMG(img, n):
    #img = img*4 - 2*scipy.ndimage.gaussian_filter(img,sigma=1)
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
 
    sift = cv.SIFT_create()
    k,d = sift.detectAndCompute(gray, None)
    output_image = cv.drawKeypoints(gray, k, 0, (255, 0, 0), 
                                 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
    
    plt.imshow(output_image)
    plt.show()

    return k,d,img

def computePairs(des_1,des_2,t):
    return np.where(distance.cdist(des_1,des_2,'sqeuclidean')< t)

def Ransac(N, p, t, k1, k2):
    n_p = p.shape[1]
    ret_ct = -1
    d_b = 0
    for x in range(N):
        ran = np.random.randint(n_p, size=4)
        A = hTrans(p, ran, k1, k2)
        U,s,V = np.linalg.svd(A)
        trans = V[-1].reshape((3,3))
        inliers, ct, d = getInliers(trans,p,k1,k2,t)
        if (ct > ret_ct) or (ret_ct == -1):
            d_b = d
            ret_ct = ct
            ret_t = trans
            ret_in = inliers
    print(ret_ct, d_b)
    return np.array(ret_in), np.array(ret_t)

def getInliers(trans,p, k1,k2, t):
    ans = []
    dists = 0
    count = 0
    for x in p.T:

        i = x[0]
        j = x[1]
        xy = np.array([k1[i].pt[0], k1[i].pt[1],1])
        xy_h = trans@xy
        xy_h = xy_h/xy_h[2]
        dist = (xy_h[0]-k2[j].pt[0])**2 + (xy_h[1]-k2[j].pt[1])**2
        if dist < t:
            dists += dist
            count += 1
            ans += [[k1[i].pt[0],k1[i].pt[1],k2[j].pt[0],k2[j].pt[1]]]
    return ans, count, dists

def hTrans(p, ran, k1, k2):
    A = np.empty([12,9])
    i = 0
    for idx in ran:
        pt1  = k1[p[0,idx]].pt
        pt2  = k2[p[1,idx]].pt

        A[i] = np.array([0,0,0, -pt1[0],-pt1[1],-1, pt2[1]*pt1[0],pt2[1]*pt1[1],pt2[1]])
        i+= 1

        A[i] = [pt1[0],pt1[1],1, 0,0,0, -pt2[0]*pt1[0],-pt2[0]*pt1[1],-pt2[0]] 
        i+= 1

        A[i] = [-pt2[1]*pt1[0],-pt2[1]*pt1[1],-pt2[1], pt2[0]*pt1[0],pt2[0]*pt1[1],pt2[0], 0,0,0]
        i+= 1
    
    return A

def cropBorder(npArray,val):
    idx = np.where(npArray > val)
    b_y = [min(idx[0]),max(idx[0])]
    b_x = [min(idx[1]),max(idx[1])]
    return npArray[b_y[0]:b_y[1],b_x[0]:b_x[1]], b_x[0], b_y[0]

def ortog(xs,ys,row,col):
    A = np.empty([12,9])
    i = 0

    if col > row:
        x1 = xs[0]-xs[1]
        y1 = ys[0]-ys[1]

        newc = (x1**2+y1**2)**(1/2)
        unit = newc/col
        newr = unit*row
    else:
        x1 = xs[0]-xs[2]
        y1 = ys[0]-ys[2]

        newr = (x1**2+y1**2)**(1/2)
        unit = newr/row
        newc = unit*col

    nx = [xs[0],xs[0],xs[0]+newr,xs[0]+newr]
    ny = [ys[0],ys[0]+newc,ys[0],ys[0]+newc]

    for idx in range(4):
        pt1  = [nx[idx],ny[idx]]
        pt2  = [xs[idx],ys[idx]]

        A[i] = np.array([0,0,0, -pt1[0],-pt1[1],-1, pt2[1]*pt1[0],pt2[1]*pt1[1],pt2[1]])
        i+= 1

        A[i] = [pt1[0],pt1[1],1, 0,0,0, -pt2[0]*pt1[0],-pt2[0]*pt1[1],-pt2[0]] 
        i+= 1

        A[i] = [-pt2[1]*pt1[0],-pt2[1]*pt1[1],-pt2[1], pt2[0]*pt1[0],pt2[0]*pt1[1],pt2[0], 0,0,0]
        i+= 1

    U,s,V = np.linalg.svd(A)
    trans = V[-1].reshape((3,3))
    return trans, unit

def get_top_and_bottom_coordinates(im):
    """
    For a specific object, prompts user to record the top coordinate and the bottom coordinate in the image.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        obj: string, object name
    Returns:
        coord: np.ndarray of shape (3, 2)
            where coord[:, 0] is the homogeneous coordinate of the top of the object and coord[:, 1] is the homogeneous
            coordinate of the bottom
    """
    plt.figure()
    plt.imshow(im)

    print('Click on the top-left coordinate')
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x1, y1 = clicked[0]
    # Uncomment this line to enable a vertical line to help align the two coordinates
    # plt.plot([x1, x1], [0, im.shape[0]], 'b')
    print('Click on the bottom-left coordinate')
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x2, y2 = clicked[0]

    print('Click on the top-right coordinate')
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x3, y3 = clicked[0]
    # Uncomment this line to enable a vertical line to help align the two coordinates
    # plt.plot([x1, x1], [0, im.shape[0]], 'b')
    print('Click on the bottom-right coordinate')
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x4, y4 = clicked[0]

    while True:
        rowlen = input("Enter dist of top-left to top-right: ")
        if rowlen.isnumeric():
            rowlen = float(rowlen)
            break
    
    while True:
        collen = input("Enter dist of top-left to bottom-left: ")
        if collen.isnumeric():
            collen = float(collen)
            break
    
    

    plt.plot([x1, x2], [y1, y2], 'b')

    return np.array([x1, x2, x3, x4]), np.array([y1, y2, y3, y4]), rowlen,collen

def plot_dimensions(ax_d, n_img, s_img, xdim, ydim, ox, oy, upp):
    
    ax_d.imshow(n_img, cmap='gray')
    ax_d.plot(ox, oy, 'ro') 
    ax_d.plot([ox,xdim], [oy,oy], [ox,ox], [oy,ydim], color="white", linewidth=3) 
    ax_d.plot([ox,xdim], [ydim,ydim], [xdim,xdim], [oy,ydim], color="white", linewidth=1, linestyle='dashed')
    ax_d.text((ox+xdim)/2,oy,str(round(abs(ox-xdim)/upp,3))+" units", verticalalignment='bottom', color="white")
    ax_d.text(ox,(oy+ydim)/2, str(round(abs(oy-ydim)/upp,3))+" units", color="white")
    
    ax_d.plot([xdim,xdim+s_img.shape[1]], [ydim+s_img.shape[0],ydim+s_img.shape[0]], [xdim+s_img.shape[1],xdim+s_img.shape[1]], [ydim,ydim+s_img.shape[0]], color="red", linewidth=1)
    ax_d.text(xdim+(s_img.shape[1])/2,ydim+s_img.shape[0],str(round(s_img.shape[1]/upp,3))+" units", verticalalignment='top', horizontalalignment='center', color="red")
    ax_d.text(xdim+s_img.shape[1],ydim+s_img.shape[0]/2, str(round(s_img.shape[0]/upp,3))+" units", color="red")

    plt.show() 

if ("_scr" != pic1[-8:-4]): 
    img1 = np.array(Image.open(pic1).convert("RGB")) 
    xs,ys,row,col = get_top_and_bottom_coordinates(img1)
    orto_h, upp = ortog(xs,ys,row,col) 
    PT = skimage.transform.ProjectiveTransform(matrix=orto_h)
    scr = np.array(skimage.transform.warp(img1,PT, output_shape=(img1.shape[0], img1.shape[1])))*255
    scr = scr.astype(np.uint8)

    fname = pic1[:-4]+"_scr.jpg"
    skimage.io.imsave(fname, scr)
    kp_1, des_1, img1 = getSIFTKPfromIMG(scr, 1000)

    dims = np.array([xs,ys,[row,col,upp,0]])
    fname = pic1[:-4]+"_info.txt"
    np.savetxt(fname,dims)

    img1 = img1[:, :, ::-1]

else:
    kp_1, des_1, img1 = getSIFTKP(pic1, 1000)
    dims = np.loadtxt(pic1[:-8]+"_info.txt")





kp_2, des_2, img2 = getSIFTKP(pic2, 1000)

t_pairs = 30000
pairs = np.array(computePairs(des_1,des_2,t_pairs))
computed_inliers, h_t = Ransac(1000,pairs,50,kp_1,kp_2)


PT = skimage.transform.ProjectiveTransform(matrix=h_t)
warped = skimage.transform.warp(img2,PT, output_shape=(img1.shape[0], img1.shape[1]))


#print(n_img[295:298,800:810])
n_img = np.where(warped != 0 , warped,img1/255)
n_img = n_img[:, :, ::-1]
crop, posx,posy = cropBorder(warped*255,1)

# #print(warped.shape,img1.shape,n_img.shape, n_img[395:398,800:810])
fname = pic2[:-4] + "_output.jpg"
data =  Image.fromarray((n_img*255).astype(np.uint8))
data.save(fname) 

fig_d, ax_d = plt.subplots(figsize=(20,10))
print(dims[0,0],dims[1,0],dims[2,2])
plot_dimensions(ax_d, n_img, crop, posx, posy,dims[0,0],dims[1,0],dims[2,2])

fig, ax = plt.subplots(figsize=(20,10))
#plot_inlier_matches(ax, img1, img2, computed_inliers)





  