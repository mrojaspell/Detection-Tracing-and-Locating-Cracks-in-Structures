# --- Imports ---
import numpy as np
import scipy.ndimage
import skimage
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy
from pylab import *
from scipy import signal
from scipy import *
from PIL import Image
from skimage import morphology, img_as_ubyte
from skimage.morphology import remove_small_objects

# --- User-defined image paths ---
pic1 = 'crack_images/ground.jpg'
pic2 = 'small_crack/ground.jpg'

# --- Functions from the first code ---

def getSIFTKP(file, n):
    img = cv.imread(file)
    # Convert BGR to RGB for consistency with matplotlib and skimage
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    sift = cv.SIFT_create()
    k,d = sift.detectAndCompute(gray, None)
    output_image = cv.drawKeypoints(img_rgb, k, None, (255, 0, 0),
                                    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
    plt.figure("SIFT Keypoints")
    plt.imshow(output_image)
    plt.title("SIFT Keypoints for " + file)
    plt.show()
    return k,d,img_rgb

def getSIFTKPfromIMG(img, n):
    # img is assumed already in RGB
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    sift = cv.SIFT_create()
    k,d = sift.detectAndCompute(gray, None)
    output_image = cv.drawKeypoints(img, k, None, (255, 0, 0),
                                    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
    plt.figure("SIFT Keypoints from IMG")
    plt.imshow(output_image)
    plt.title("SIFT Keypoints from Input IMG")
    plt.show()
    return k,d,img

def computePairs(des_1,des_2,t):
    return np.where(distance.cdist(des_1,des_2,'sqeuclidean')< t)

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

def Ransac(N, p, t, k1, k2):
    n_p = p.shape[1]
    ret_ct = -1
    d_b = 0
    ret_t = None
    ret_in = None
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
    print("Best Inliers Count:", ret_ct, "Total Distance:", d_b)
    print("Homography:\n", ret_t)
    return np.array(ret_in), np.array(ret_t)

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
    plt.figure()
    plt.imshow(im)
    plt.title("Select Points on the Large Image")

    print('Click on the top-left coordinate')
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x1, y1 = clicked[0]

    print('Click on the bottom-left coordinate')
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x2, y2 = clicked[0]

    print('Click on the top-right coordinate')
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x3, y3 = clicked[0]

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
    plt.show()
    return np.array([x1, x2, x3, x4]), np.array([y1, y2, y3, y4]), rowlen,collen

def plot_dimensions(ax_d, n_img, s_img, xdim, ydim, ox, oy, upp, fname_dim):
    ax_d.imshow(n_img)  # preserve color
    ax_d.plot(ox, oy, 'ro') 
    ax_d.plot([ox,xdim], [oy,oy], [ox,ox], [oy,ydim], color="white", linewidth=3) 
    ax_d.plot([ox,xdim], [ydim,ydim], [xdim,xdim], [oy,ydim], 
              color="white", linewidth=1, linestyle='dashed')
    ax_d.text((ox+xdim)/2,oy,str(round(abs(ox-xdim)/upp,3))+" units", 
              verticalalignment='top', color="white")
    ax_d.text(ox,(oy+ydim)/2, str(round(abs(oy-ydim)/upp,3))+" units", color="white")

    ax_d.plot([xdim,xdim+s_img.shape[1]], [ydim+s_img.shape[0],ydim+s_img.shape[0]],
              [xdim+s_img.shape[1],xdim+s_img.shape[1]], [ydim,ydim+s_img.shape[0]], 
              color="red", linewidth=1)
    ax_d.text(xdim+(s_img.shape[1])/2,ydim+s_img.shape[0],str(round(s_img.shape[1]/upp,3))+" units", 
              verticalalignment='top', horizontalalignment='center', color="red")
    ax_d.text(xdim+s_img.shape[1],ydim+s_img.shape[0]/2, str(round(s_img.shape[0]/upp,3))+" units", color="red")

    plt.savefig(fname_dim)
    plt.show()

# --- Crack detection function ---
def detect_cracks(gray, block_size, C, min_size, morph_kernel, area_thresh, length_thresh):
    gray_blurred = cv.GaussianBlur(gray, (5, 5), 0)
    binary = cv.adaptiveThreshold(gray_blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY_INV, block_size, C)

    binary_bool = binary.astype(bool)
    binary_cleaned = remove_small_objects(binary_bool, min_size=min_size)
    binary_cleaned = (binary_cleaned.astype(np.uint8) * 255)

    if morph_kernel is not None:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, morph_kernel)
        binary_cleaned = cv.morphologyEx(binary_cleaned, cv.MORPH_OPEN, kernel)

    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    binary_closed = cv.morphologyEx(binary_cleaned, cv.MORPH_CLOSE, close_kernel)

    border = 5
    binary_closed[:border, :] = 0
    binary_closed[-border:, :] = 0
    binary_closed[:, :border] = 0
    binary_closed[:, -border:] = 0

    skeleton = morphology.skeletonize(binary_closed // 255)
    skeleton = img_as_ubyte(skeleton)

    contours, _ = cv.findContours(skeleton, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    base_length_threshold = 50
    max_aspect_ratio = 2000
    max_thickness_ratio = 500

    valid_contours = []
    for contour in contours:
        length = cv.arcLength(contour, closed=False)
        area = cv.contourArea(contour)

        if length < length_thresh:
            continue
        if area < area_thresh:
            continue

        x, y, w, h = cv.boundingRect(contour)
        if h == 0 or w == 0:
            continue
        aspect_ratio = max(w/h, h/w)
        thickness_ratio = area / (length + 1e-5)

        if aspect_ratio > max_aspect_ratio:
            continue
        if thickness_ratio > max_thickness_ratio:
            continue

        valid_contours.append(contour)

    return valid_contours

# --- Main Logic ---

# Step 1: Load and prepare img1 (as RGB)
img1_pil = Image.open(pic1).convert("RGB")
img1 = np.array(img1_pil)  # RGB
print("img1 shape:", img1.shape)

# If we don't have the rectified "_scr" version of pic1, do the interactive step:
try:
    dims = np.loadtxt(pic1[:-4]+"_info.txt")
except:
    img1 = np.array(Image.open(pic1).convert("RGB")) 
    xs,ys,row,col = get_top_and_bottom_coordinates(img1)
    orto_h, upp = ortog(xs,ys,row,col) 
    PT = skimage.transform.ProjectiveTransform(matrix=orto_h)
    scr = np.array(skimage.transform.warp(img1,PT, output_shape=(img1.shape[0], img1.shape[1])))*255
    scr = scr.astype(np.uint8)

    dims = np.array([xs,ys,[row,col,upp,0]])
    dims = np.hstack((dims,orto_h))
    fname = pic1[:-4]+"_info.txt"
    np.savetxt(fname,dims)

    img1 = img1[:, :, ::-1]


kp_1, des_1, img1 = getSIFTKP(pic1, 1000)

# Step 2: Load and prepare img2 (as RGB)
img2_bgr = cv.imread(pic2)
if img2_bgr is None:
    raise FileNotFoundError(f"Could not load {pic2}")
img2 = cv.cvtColor(img2_bgr, cv.COLOR_BGR2RGB)
print("img2 shape:", img2.shape)

# Compute SIFT for img2
kp_2, des_2, img2_rgb = getSIFTKPfromIMG(img2, 1000)

# Compute pairs and run RANSAC to find homography
t_pairs = 30000
pairs = np.array(computePairs(des_1,des_2,t_pairs))
computed_inliers, h_t = Ransac(1000,pairs,50,kp_1,kp_2)

# Warp img2 into img1's coordinate system using full img1 size
img2_float = img2.astype(float)/255.0
PT = skimage.transform.ProjectiveTransform(matrix=h_t)
#warped = skimage.transform.warp(img2_float, PT, output_shape=(img1.shape[0], img1.shape[1]))
#print("warped shape:", warped.shape)

# Create a mask for where the smaller image would appear
#warped_mask = np.any(warped > 0.01, axis=-1)  # Only overwrite where warped is >0.01 in any channel
n_img = img1.astype(float)/255.0

# Save intermediate images for debugging
#skimage.io.imsave("debug_warped.jpg", (warped*255).astype(np.uint8))

#fname = pic2[:-4] + "_output.jpg"
#data = (n_img*255).astype(np.uint8)
#Image.fromarray(data).save(fname)

#fig_d, ax_d = plt.subplots(figsize=(20,10))
#print("dims:", dims[0,0], dims[1,0], dims[2,2])
#plot_dimensions(ax_d, n_img, warped, 0, 0, dims[0,0], dims[1,0], dims[2,2])

# --- Crack Detection on pic2 (the smaller image) ---
gray_small = cv.cvtColor(img2_bgr, cv.COLOR_BGR2GRAY)

block_sizes = [15, 25, 35, 45, 55]
C_values = [5, 7, 9]
min_size_candidates = [50]
morph_kernels = [(3,3)]
area_threshold_values = [0]

base_length_threshold = 50

best_crack = None
best_length = 0
best_params = None

for bs in block_sizes:
    for C_val in C_values:
        for ms in min_size_candidates:
            for mk in morph_kernels:
                for area_thresh in area_threshold_values:
                    length_thresh = base_length_threshold
                    contours = detect_cracks(
                        gray=gray_small, 
                        block_size=bs, 
                        C=C_val, 
                        min_size=ms, 
                        morph_kernel=mk,
                        area_thresh=area_thresh, 
                        length_thresh=length_thresh
                    )

                    if contours:
                        # Sort by length
                        contours.sort(key=lambda cnt: cv.arcLength(cnt, False), reverse=True)
                        top_crack = contours[0]
                        top_length = cv.arcLength(top_crack, closed=False)

                        if top_length > best_length:
                            best_length = top_length
                            best_crack = top_crack
                            best_params = (bs, C_val, ms, mk, area_thresh, length_thresh)

if best_crack is not None:
    length = cv.arcLength(best_crack, closed=False)
    M = cv.moments(best_crack)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv.boundingRect(best_crack)
        cx, cy = x + w//2, y + h//2

    print("Found a suitable crack.")
    print(f"Parameters used:\n  block_size={best_params[0]}, C={best_params[1]}, min_size={best_params[2]}, morph_kernel={best_params[3]}, area_thresh={best_params[4]}, length_thresh={best_params[5]}")
    print(f"Crack position (centroid in small image): ({cx}, {cy}), Length: {length:.2f} pixels")

    # Create a black canvas the size of img2 and draw the crack in white
    canvas_small = np.zeros_like(img2_bgr)  # black background
    cv.drawContours(canvas_small, [best_crack], -1, (255,255,255), 2)  # white crack line

    # Warp this black/white canvas into the large image space using the same homography
    canvas_small_float = canvas_small.astype(float)/255.0
    warped_crack = skimage.transform.warp(canvas_small_float, PT, output_shape=(img1.shape[0], img1.shape[1]))
    warped_crack8 = (warped_crack*255).astype(np.uint8)

    # Instead of placing black and white image, we only overlay the white crack line
    # Create a mask where the crack line is white
    # The crack line is (255,255,255), so we check if all channels > 200, for example:
    line_mask = np.all(warped_crack8 > 200, axis=-1)

    # Overlay only the crack line onto n_img
    # n_img is currently in float [0,1], set line pixels to white [1,1,1]
    n_img[line_mask] = [1.0, 1.0, 1.0]

    overlay = (n_img*255).astype(np.uint8)
    overlay = overlay[:, :, ::-1]
    cv.imshow("Crack on Large Image", overlay)
    fname = pic2[:-4] + "_output.jpg"
    cv.imwrite(fname, overlay)
    cv.waitKey(0)
    cv.destroyAllWindows()

    fig_d, ax_d = plt.subplots(figsize=(20,10))
    print("dims:", dims[0,0], dims[1,0], dims[2,2])
    crop, posx,posy = cropBorder(warped_crack*255,1)
    fname_dim = pic2[:-4] + "_output_dim.jpg"
    plot_dimensions(ax_d, n_img, crop, posx, posy, dims[0,0], dims[1,0], dims[2,2], fname_dim)
else:
    print("No suitable crack found after testing all parameter combinations.")

