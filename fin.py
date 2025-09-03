import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- 1. HISTOGRAM ----------------
def compute_histogram(img):
    print("[STEP] Computing histogram...")
    h, w = img.shape
    hist = np.zeros(256, dtype=int)
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    print("[INFO] Histogram computed. Length =", len(hist))
    print("[INFO] First 10 values:", hist[:10])
    return hist

# ---------------- 2. TRANSFORMATION ----------------
def contrast_stretch(img):
    r_min, r_max = np.min(img), np.max(img)
    print(f"[STEP] Contrast stretching: min={r_min}, max={r_max}")
    stretched = ((img - r_min) / (r_max - r_min)) * 255
    return stretched.astype(np.uint8)

def negative(img):
    print("[STEP] Applying negative transformation...")
    return 255 - img

# ---------------- 3. EDGE DETECTION ----------------
def sobel_edge(img):
    print("[STEP] Performing Sobel edge detection...")
    gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
    gy_kernel = np.array([[-1, -2, -1],
                          [0,  0,  0],
                          [1,  2,  1]])
    h, w = img.shape
    edge_img = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            region = img[i-1:i+2, j-1:j+2]
            gx = np.sum(region * gx_kernel)
            gy = np.sum(region * gy_kernel)
            edge_val = np.sqrt(gx**2 + gy**2)
            edge_img[i, j] = np.clip(edge_val, 0, 255)
    print("[INFO] Sobel edge detection done.")
    return edge_img

# ---------------- MANUAL CANNY EDGE DETECTION ----------------
def canny_edge(img, low_thresh=50, high_thresh=150):
    print("[STEP] Performing manual Canny edge detection...")

    # 1. Gaussian smoothing
    print("[INFO] Step 1: Gaussian smoothing...")
    kernel = cv2.getGaussianKernel(5, 1)
    gaussian = cv2.filter2D(img, -1, kernel @ kernel.T)

    # 2. Compute gradients (Sobel)
    print("[INFO] Step 2: Gradient computation (Sobel)...")
    gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
    gy_kernel = np.array([[-1, -2, -1],
                          [0,  0,  0],
                          [1,  2,  1]])
    h, w = img.shape
    G = np.zeros((h, w))
    theta = np.zeros((h, w))
    for i in range(1, h-1):
        for j in range(1, w-1):
            region = gaussian[i-1:i+2, j-1:j+2]
            gx = np.sum(region * gx_kernel)
            gy = np.sum(region * gy_kernel)
            G[i, j] = np.sqrt(gx**2 + gy**2)
            theta[i, j] = np.arctan2(gy, gx)

    # 3. Non-maximum suppression
    print("[INFO] Step 3: Non-maximum suppression...")
    Z = np.zeros((h, w))
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1, h-1):
        for j in range(1, w-1):
            q = 255
            r = 255
            # Horizontal edge
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = G[i, j+1]
                r = G[i, j-1]
            # Diagonal 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            # Vertical
            elif (67.5 <= angle[i,j] < 112.5):
                q = G[i+1, j]
                r = G[i-1, j]
            # Diagonal 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = G[i-1, j-1]
                r = G[i+1, j+1]

            if G[i,j] >= q and G[i,j] >= r:
                Z[i,j] = G[i,j]
            else:
                Z[i,j] = 0

    # 4. Double threshold
    print("[INFO] Step 4: Double thresholding...")
    res = np.zeros((h,w), dtype=np.uint8)
    strong = 255
    weak = 75
    strong_i, strong_j = np.where(Z >= high_thresh)
    weak_i, weak_j = np.where((Z <= high_thresh) & (Z >= low_thresh))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # 5. Edge tracking by hysteresis
    print("[INFO] Step 5: Edge tracking by hysteresis...")
    for i in range(1, h-1):
        for j in range(1, w-1):
            if res[i,j] == weak:
                if 255 in res[i-1:i+2, j-1:j+2]:
                    res[i,j] = strong
                else:
                    res[i,j] = 0

    print("[INFO] Manual Canny edge detection done.")
    return res

# ---------------- 4. CONNECTED COMPONENTS ----------------
def connected_components(binary_img):
    print("[STEP] Computing connected components...")
    h, w = binary_img.shape
    label = 1
    labels = np.zeros((h, w), dtype=int)
    def dfs(x, y, label):
        stack = [(x, y)]
        while stack:
            i, j = stack.pop()
            if 0 <= i < h and 0 <= j < w and binary_img[i, j] == 255 and labels[i, j] == 0:
                labels[i, j] = label
                stack.extend([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])
    for i in range(h):
        for j in range(w):
            if binary_img[i, j] == 255 and labels[i, j] == 0:
                dfs(i, j, label)
                label += 1
    print(f"[INFO] Total connected components found: {label-1}")
    labels_img = (labels * (255 // label)).astype(np.uint8)
    return labels_img

# ---------------- 5. THRESHOLDING ----------------
def binary_threshold(img, thresh=127):
    print(f"[STEP] Applying binary threshold with T={thresh}...")
    return np.where(img > thresh, 255, 0).astype(np.uint8)

def otsu_threshold(img):
    print("[STEP] Computing Otsu's threshold...")
    hist = compute_histogram(img)
    total = img.size
    sum_total = np.sum([i * hist[i] for i in range(256)])
    sum_b, w_b, w_f, var_max, threshold = 0, 0, 0, 0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t
    print(f"[INFO] Otsu's threshold: {threshold}")
    return binary_threshold(img, threshold)

def multiple_threshold(img, thresholds=[85, 170]):
    print(f"[STEP] Applying multiple thresholds: {thresholds}...")
    res = np.zeros_like(img)
    res[img < thresholds[0]] = 0
    res[(img >= thresholds[0]) & (img < thresholds[1])] = 127
    res[img >= thresholds[1]] = 255
    return res

# ---------------- 6. SAMPLING & QUANTIZATION ----------------
def sampling(img, factor=2):
    print(f"[STEP] Sampling image with factor={factor}...")
    return img[::factor, ::factor]

def quantization(img, levels=16):
    print(f"[STEP] Quantizing image with {levels} levels...")
    return (img // (256 // levels)) * (256 // levels)

# ---------------- 7. FIRST / ZERO ORDER HOLD ----------------
def zero_order_hold(img, scale=2):
    print(f"[STEP] Scaling image using zero-order hold by {scale}...")
    h, w = img.shape
    out = np.zeros((h*scale, w*scale), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = img[i, j]
    return out

def first_order_hold(img, scale=2):
    print(f"[STEP] Scaling image using first-order hold by {scale}...")
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

# ---------------- 8. BASIC IMAGE OPS ----------------
def basic_ops(img):
    print("[STEP] Performing basic image operations (flip, rotate, crop)...")
    flipped = np.flipud(img)
    rotated = np.rot90(img)
    h, w = img.shape
    cropped = img[h//4:3*h//4, w//4:3*w//4]
    return flipped, rotated, cropped

# ---------------- 9. HISTOGRAM EQUALIZATION ----------------
def histogram_equalization(img):
    print("[STEP] Performing histogram equalization...")
    hist = compute_histogram(img)
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    equalized = cdf_normalized[img]
    print("[INFO] Histogram equalization done.")
    return equalized.astype(np.uint8)

# ---------------- 10. ARITHMETIC / LOGICAL OPERATIONS ----------------
def add_images(img1, img2):
    print("[STEP] Adding two images...")
    return np.clip(img1.astype(int) + img2.astype(int), 0, 255).astype(np.uint8)

def and_images(img1, img2):
    print("[STEP] AND operation between images...")
    return np.bitwise_and(img1, img2)

def or_images(img1, img2):
    print("[STEP] OR operation between images...")
    return np.bitwise_or(img1, img2)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    img = cv2.imread(r"C:/Users/Samveda.SAMVEDA/Downloads/sample.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Error: Image not found. Check the path.")
        exit()

    # Apply all operations
    neg = negative(img)
    sobel = sobel_edge(img)
    canny = canny_edge(img)
    binary = binary_threshold(img)
    otsu = otsu_threshold(img)
    multi_thresh = multiple_threshold(img)
    labels = connected_components(binary)
    eq = histogram_equalization(img)
    zoh = zero_order_hold(img, 2)
    foh = first_order_hold(img, 2)
    sampled = sampling(img, 4)
    quantized = quantization(img, 8)
    flipped, rotated, cropped = basic_ops(img)
    stretched = contrast_stretch(img)

    # Display all results
    images = [
        ("Original", img), ("Negative", neg), ("Sobel Edge", sobel), ("Canny Edge", canny),
        ("Binary Thresh", binary), ("Otsu Thresh", otsu), ("Multi Thresh", multi_thresh),
        ("Connected Components", labels), ("Hist Equalized", eq), ("Contrast Stretch", stretched),
        ("Zero Order Hold", zoh), ("First Order Hold", foh), ("Sampled", sampled),
        ("Quantized", quantized), ("Flipped", flipped), ("Rotated", rotated), ("Cropped", cropped)
    ]

    plt.figure(figsize=(18, 14))
    for i, (title, im) in enumerate(images, 1):
        plt.subplot(5, 4, i)
        plt.imshow(im, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # ---------------- DISPLAY IMAGES INDIVIDUALLY ----------------
    for title, im in images:
        plt.figure(figsize=(6, 6))
        plt.imshow(im, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()
