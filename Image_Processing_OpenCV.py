import cv2 as cv
import numpy as np

# Gives numpy array of histogram of 8-bit igray level image
def get_hist(gray):
    
    (rows,cols) = gray.shape

    # Calculating histogram of the given gray level image
    hist = np.zeros(256,dtype=np.uint32)

    for i in range(rows):
        for j in range(cols):
            hist[gray[i,j]] += 1

    return hist

# Histogram Equalization for 8-bit Gray Level Image Martix 'gray'
def hist_eq(gray):

    (rows,cols) = gray.shape
    total_pixels = rows*cols

    # hist stands for Histogram
    hist = get_hist(gray)

    # Calculating Cumulative distribution function
    cdf = np.zeros(256,dtype=np.float32)

    # Calculation CDF of the given histogram
    cdf_prev = 0
    for i in range(256):
        cdf[i] = cdf_prev + float(hist[i]/total_pixels)
        cdf_prev = cdf[i]

    # Maps old gray level value to the new gray level values
    equi = np.zeros(256,dtype=np.uint8)

    for i in range(256):
        equi[i] = round(cdf[i]*255)

    # Changing the gray level matrix in-place
    for i in range(rows):
        for j in range(cols):
            gray[i,j] = equi[gray[i,j]]

    # free memory
    del hist
    del cdf
    del equi

def fill_window(gray,x,y,windowSize):
    
    (rows,cols) = gray.shape
    window = np.zeros(windowSize*windowSize,dtype='uint8')
    halfSize = windowSize//2
    curr = 0

    # Uses Zero Padding
    for i in range(x-halfSize,x+halfSize+1):
        for j in range(y-halfSize,y+halfSize+1):
            if i<0 or i>=rows or j<0 or j>=cols:
                window[curr] = 0        # Zero Padding
            else:
                window[curr] = gray[i,j]
            curr += 1
        
    return window

# Apply this adaptive medain filter to reduce salt and pepper noise
def adaptive_median_filter(gray):

    def adaptive_median_filter_pixel(gray,x,y,windowSize):

        S_max = 21      # Maximum possible window size

        window = fill_window(gray,x,y,windowSize)

        z_min = int(np.min(window))
        z_max = int(np.max(window))
        z_median = int(np.median(window))
        z_xy = int(gray[x,y])

        # Level A
        a1 = z_median - z_min
        a2 = z_median - z_max

        if a1 > 0 and a2 < 0:
            # Level B
            b1 = z_xy - z_min
            b2 = z_xy - z_max

            if b1 > 0 and b2 < 0:
                res = z_xy
            else:
                res = z_median
        else:
            # Not getting a good value (window is small to analize)
            # Increasing Window Size
            windowSize += 2
            if windowSize > S_max:
                res = z_xy
            else:
                res = adaptive_median_filter_pixel(gray,x,y,windowSize)

        del window

        return res

    (rows,cols) = gray.shape

    # We will return this new image as output on this function
    filtered_img = np.zeros((rows,cols),dtype='uint8')

    # applying the adaptive median filter on each pixel
    for i in range(rows):
        for j in range(cols):
            filtered_img[i,j] = adaptive_median_filter_pixel(gray,i,j,3)        # initial window size in convolution as 3

    return filtered_img

# Returns a new image (matrix) with gaussian filter applied, helps to smooth the image and reduces noise
def gaussian_filter(gray):

    (rows,cols) = gray.shape

    # We will return this new image as output on this function
    filtered_img = np.zeros((rows,cols),dtype='uint8')

    # Representing the filter
    filter = np.array([0.0625,0.125,0.0625,0.125,0.250,0.125,0.0625,0.125,0.0625],dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            window = fill_window(gray,i,j,3)
            output_val = 0
            for x in range(9):
                output_val += filter[x] * float(window[x])
            
            filtered_img[i,j] = round(output_val)
    
    return filtered_img



# NOT WORKING

'''
# Adaptive local Mean Filter is used to reduce gaussian noise, it return a new image (matrix)
def adaptive_mean_filter(gray):
    
    # f(x,y) = g(x,y) - (sigma_n^2 / sigma_l^2)*(g(x,y) - m_l)

    # sigma_l^2 : local variance of local region
    # m_l : local mean
    # sigma_n^2 : variance of overall noise
    # g(x,y) : pixel value at position (x,y) in gray

    # var_n == sigma_n^2
    # var_l == sigma_l^2

    (rows,cols) = gray.shape

    # We will return this new image as output on this function
    filtered_img = np.zeros((rows,cols),dtype='uint8')

    var_n = gray.var()/255

    if round(var_n) == 0:
        return gray
    
    # Applying the formula
    for i in range(rows):
        for j in range(cols):
            window = fill_window(gray,i,j,3)
            window.astype(np.float32)

            for x in range(9):
                window[x] = window[x]/255

            var_l = window.var()
            m_l = np.mean(window)

            if round(var_l) == 0:
                filtered_img[i,j] = gray[i,j]
                continue

            temp = gray[i,j]/255 - ((var_n / var_l) * (gray[i,j]/255 - m_l))
            temp = gray[i,j] if (temp<0 or temp>1) else temp

            filtered_img[i,j] = round(temp*255)

    return filtered_img
'''

# Shrapen The Image
def spatial_filter(gray,choice = 1):

    if choice!=1 and choice!=2:
        raise ValueError("> spatial_filter() should have choice as 1 or 2")

    (rows, cols) = gray.shape

    filter = (
                np.array([[0,1,0],[1,-4,1],[0,1,0]]),
                np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
            )

    filtered_image = cv.filter2D(src = gray, ddepth = 0, kernel = filter[choice-1])

    #filtered_image = cv.add(filtered_image,gray)
    #filtered_image = cv.subtract(gray,filtered_image)

    return filtered_image

# Used for Edge Detection, return a new image
def sobel_filter(gray):

    filter_l = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    filter_r = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    img1 = cv.filter2D(src = gray, ddepth = 0, kernel = filter_l)
    img2 = cv.filter2D(src = gray, ddepth = 0, kernel = filter_r)

    filtered_image = cv.add(img1,img2)

    return filtered_image

# Uses filter with
def line_detection(gray,choice = 1):

    if choice<1 or choice>4:
        raise ValueError("> spatial_filter() should have choice be in (1,2,3,4)")

    filter = (
                np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]]),     # Horizontal
                np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]]),     # Vertical
                np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]]),     # 45 degree
                np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])      # -45 degree    
            )
    
    img = cv.filter2D(src = gray, ddepth = 0, kernel = filter[choice])

    return img

# Divide image into Background and Foreground
def otsu_thresholding(gray):

    th, binary_img = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)

    return binary_img

def boundary_extraction(binary_img):

    eroded = cv.erode(src = binary_img, kernel = (3,3), iterations = 2)
    dialated = cv.dilate(src = eroded, kernel = (3,3), iterations = 2)
    dialated = cv.dilate(src = eroded, kernel = (3,3), iterations = 2)
    eroded = cv.erode(src = dialated, kernel = (3,3), iterations = 2)

    del dialated

    kernel = np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]],dtype='uint8')

    dialated = cv.dilate(src = eroded, kernel = kernel, iterations = 1)
    boundary = cv.subtract(dialated, eroded)

    del dialated
    del eroded

    return boundary

def main():
    # Reading Image using cv.imread()
    img = cv.imread("OpenCV Python\img3.jpg")
    #img = cv.imread("OpenCV Python\image.png")

    if img is None:
        raise ValueError("File Path specified doesn't exist.")

    # Python have BGR Format (not RGB)
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]

    # converting to gray level image
    gray = 0.299*R + 0.587*G + 0.114*B
    gray = gray.astype(np.uint8)

    # Using Built-in
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    del B
    del R
    del G
    del img

    #cv.imshow("Image",img)
    cv.imshow("Gray",gray)
    #cv.waitKey(0)

    (rows,cols) = gray.shape
    total_pixels = rows * cols

    #hist_eq(gray)

    #cv.imshow("Hist Equi",gray)

    #filtered_img = adaptive_median_filter(gray)

    #filtered_img = gaussian_filter(gray)

    #filtered_img = adaptive_mean_filter(gray)

    #filtered_img = spatial_filter(gray,1)

    #filtered_img = sobel_filter(gray,2)

    binary_img = otsu_thresholding(gray)
    filtered_img = boundary_extraction(binary_img)

    cv.imshow("Filtered Image",filtered_img)
    cv.waitKey(0)

    del binary_img
    del filtered_img
    del gray

if __name__ == "__main__":
    main()