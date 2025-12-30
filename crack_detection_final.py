import cv2
import numpy as np
from skimage import morphology, img_as_ubyte
from skimage.morphology import remove_small_objects

image = cv2.imread('small_crack/crack6_small.jpg', cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("Image not found. Check the path.")

H, W = image.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Parameter sets to try:
block_sizes = [15, 25, 35, 45, 55]
C_values = [5, 7, 9]
min_size_candidates = [50]#, 100, 200, 300, 500]
morph_kernels = [(3,3)]#, (5,5)]
area_threshold_values = [0]#, 10, 20, 30, 50]

base_length_threshold = 50
max_aspect_ratio = 2000
max_thickness_ratio = 500
border = 5

def detect_cracks(gray, block_size, C, min_size, morph_kernel, area_thresh, length_thresh):
    # Slight blur
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, C)

    # Remove small objects
    binary_bool = binary.astype(bool)
    binary_cleaned = remove_small_objects(binary_bool, min_size=min_size)
    binary_cleaned = (binary_cleaned.astype(np.uint8) * 255)

    # Morphological opening if specified
    if morph_kernel is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
        binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_OPEN, kernel)

    # Morphological closing to connect nearby segments
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_closed = cv2.morphologyEx(binary_cleaned, cv2.MORPH_CLOSE, close_kernel)

    # Remove border influence
    binary_closed[:border, :] = 0
    binary_closed[-border:, :] = 0
    binary_closed[:, :border] = 0
    binary_closed[:, -border:] = 0

    # Skeletonize after closing
    skeleton = morphology.skeletonize(binary_closed // 255)
    skeleton = img_as_ubyte(skeleton)

    # Find contours
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for contour in contours:
        length = cv2.arcLength(contour, closed=False)
        area = cv2.contourArea(contour)

        if length < length_thresh:
            continue
        if area < area_thresh:
            continue

        x, y, w, h = cv2.boundingRect(contour)
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

best_crack = None
best_length = 0
best_params = None

# Try all combinations of parameters
for bs in block_sizes:
    for C_val in C_values:
        for ms in min_size_candidates:
            for mk in morph_kernels:
                for area_thresh in area_threshold_values:
                    length_thresh = base_length_threshold
                    contours = detect_cracks(
                        gray=gray, 
                        block_size=bs, 
                        C=C_val, 
                        min_size=ms, 
                        morph_kernel=mk,
                        area_thresh=area_thresh, 
                        length_thresh=length_thresh
                    )

                    if contours:
                        # Sort by length to find the longest contour
                        contours.sort(key=lambda cnt: cv2.arcLength(cnt, False), reverse=True)
                        top_crack = contours[0]
                        top_length = cv2.arcLength(top_crack, closed=False)

                        # If this crack is better (longer) than what we have found before, keep it
                        if top_length > best_length:
                            best_length = top_length
                            best_crack = top_crack
                            best_params = (bs, C_val, ms, mk, area_thresh, length_thresh)

# After finding the best crack, draw only that crack
if best_crack is not None:
    canvas = np.zeros_like(image)
    cv2.drawContours(canvas, [best_crack], -1, (255, 255, 255), 1)

    length = cv2.arcLength(best_crack, closed=False)
    M = cv2.moments(best_crack)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(best_crack)
        cx, cy = x + w//2, y + h//2

    print("Found a suitable crack.")
    print(f"Parameters used:\n  block_size={best_params[0]}, C={best_params[1]}, min_size={best_params[2]}, morph_kernel={best_params[3]}, area_thresh={best_params[4]}, length_thresh={best_params[5]}")
    print(f"Crack position (centroid): ({cx}, {cy}), Length: {length:.2f} pixels")
    cv2.imshow("Detected Cracks", canvas)
    cv2.imwrite("adaptive_crack_detection.jpg", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No suitable crack found after testing all parameter combinations.")
