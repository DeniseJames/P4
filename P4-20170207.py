
# coding: utf-8

# In[1]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
get_ipython().magic('matplotlib inline')
from scipy import signal
from scipy.signal import find_peaks_cwt
import glob
import matplotlib.gridspec as gridspec
from scipy import ndimage
import sys
from pylab import *
from skimage.io import imsave
import pylab


# In[ ]:




# In[2]:

# Define a class to receive the characteristics of each line detection
#Attributes:
#  nonzeros: If < 1000 pixels not detected
#        
#  detected: A boolean True/False value if lines are detected
 
class Line():
    def __init__(self):
        pass 
        
        
    def line_detect(self, array):
        # was the line detected in the last iteration?
        pixels = np.sum(array)
        if pixels < 3000:
            self.detected = False 
        else:
            self.detected = True
        
        
    def therest():
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = No
        ne  
        #y values for detected line pixels
        self.ally = None


# The following code to calibrate the camera is taken from the
# Udacity Lecture notes
# 

# In[3]:

-


# In[41]:

# Test undistortion on an image
img = cv2.imread('CarND-Advanced-Lane-Lines-master/camera_cal/calibration2.jpg')

img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('CarND-Advanced-Lane-Lines-master/camera_cal//test_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)


# In[5]:

#Apply the distortion correction to the raw image.
#image = mpimg.imread('CarND-Advanced-Lane-Lines-master/test_images/solidYellowLeft.jpg')
image = cv2.imread('CarND-Advanced-Lane-Lines-master/test_images/test1.jpg')


# In[6]:


def corners_unwarp(img, mtx, dist):
    # Pass in the image into this fumction
    #undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

undist = corners_unwarp(image, mtx, dist)
undist_size = (undist.shape[1],undist.shape[0])
print(undist_size )


# In[7]:


# plot the result
f, (ax1, ax2) = plt.subplots(1,2, figsize=(24,9))
f.tight_layout()
ax1.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image', fontsize = 50)
ax2.imshow(cv2.cvtColor(undist,cv2.COLOR_BGR2RGB))
ax2.set_title('Undistored Image', fontsize=50)



# In[8]:

# define 4 source points for perspective transformation
src = np.float32([[220,719],[1220,719],[750,480],[550,480]])
# define 4 destination points for perspective transformation
dst = np.float32([[240,719],[1040,719],[1040,0],[240,0]])


M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
# Warp an image using the perspective transform, M:
birdseye = cv2.warpPerspective(undist, M, (1280, 720), flags=cv2.INTER_LINEAR)

pts = np.array([[240,719],[600,450],[710,450],[1180,719]], np.int32)
pts = pts.reshape((-1,1,2))


# plot the result
f, (ax1, ax2) = plt.subplots(1,2, figsize=(24,9))
f.tight_layout()
ax1.imshow(cv2.cvtColor(undist,cv2.COLOR_BGR2RGB))
ax1.set_title('Undistored Image', fontsize = 40)
ax2.imshow(cv2.cvtColor(birdseye,cv2.COLOR_BGR2RGB))
ax2.set_title('Bird eye Image', fontsize=40)


# In[9]:

polyshape = cv2.fillPoly(np.zeros_like(undist), np.int_([pts]), (0,255, 0))
imshow(cv2.cvtColor(polyshape,cv2.COLOR_BGR2RGB))


# In[10]:

imagefromeye = cv2.warpPerspective(birdseye, Minv, (1280, 720))

# plot the result
f, (ax1, ax2) = plt.subplots(1,2, figsize=(24,9))
f.tight_layout()
ax1.imshow(cv2.cvtColor(birdseye,cv2.COLOR_BGR2RGB))
ax1.set_title('Birdseye Image', fontsize = 40)
ax2.imshow(cv2.cvtColor(imagefromeye,cv2.COLOR_BGR2RGB))
ax2.set_title('Undistored Image', fontsize=40)


# In[11]:

def color_mask(img,min_thres,max_thres):
    return cv2.inRange(img, min_thres, max_thres) / 255

def binarize_img(img, min_thres, max_thres):
    return cv2.inRange(img, min_thres, max_thres) / 255

def apply_color_mask(hsv,img,low,high):
    mask = cv2.inRange(hsv, low, high)
    res = cv2.bitwise_and(img,img, mask= mask)
    return res


# In[ ]:




# In[12]:

image_hsv = cv2.cvtColor(undist, cv2.COLOR_RGB2HSV)

# Mask out all but yellow in the image
hsv_lower_yellow  = np.array([ 0, 80, 0], dtype = "uint8")
hsv_upper_yellow = np.array([ 30, 255, 255], dtype = "uint8")
image_hsv_yellow = apply_color_mask(image_hsv,undist,hsv_lower_yellow,hsv_upper_yellow)
plt.imshow(image_hsv_yellow)


# In[13]:

sensitivity = 35
hsv_lower_white = np.array([0,0,255-sensitivity], dtype=np.uint8)
hsv_upper_white = np.array([255,sensitivity,255], dtype=np.uint8)

image_hsv_white = apply_color_mask(image_hsv,undist,hsv_lower_white,hsv_upper_white)

plt.imshow(image_hsv_white)


# In[14]:

plt.imshow(cv2.cvtColor(undist,cv2.COLOR_BGR2RGB))


# In[15]:

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F,1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F,0, 1))
    
    
    # scale to 8-bit (0-255) then convert to type=uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # create a mask of 1's where the scaled gradient magnitude
    # is > thres_min and < thres_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel>=thresh[0])&(scaled_sobel<=thresh[1])]=1
    # return this mask as the binary_output
    
    return binary_output


# In[16]:

def mag_threshold(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # TAKE THE DERIVATIVE OF x or y 
    sobelx=cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely=cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output [(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


# In[17]:

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
     # TAKE THE DERIVATIVE OF x or y 
    sobelx=cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely=cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    #use np.arctan2() to calculate direction of gradient
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)    
    # create a mask of 1's where the scaled gradient magnitude
    # is > thres_min and < thres_max
    sxbinary = np.zeros_like(dir_sobel)
    sxbinary[(dir_sobel>=thresh[0])&(dir_sobel<=thresh[1])]=1
    # return this mask as the binary_output
    binary_output = sxbinary
    return binary_output


# In[18]:

# Choose a Sobel kernel size
ksize = 15 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(birdseye, orient='x', sobel_kernel=ksize, thresh=(20, 255))
grady = abs_sobel_thresh(birdseye, orient='y', sobel_kernel=ksize, thresh=(0, 80))

# plot the result
f, (ax1, ax2) = plt.subplots(1,2, figsize=(24,9))
f.tight_layout()
ax1.imshow(gradx, cmap='gray')
ax1.set_title('Absolute gradx Threshold Image', fontsize = 40)
ax2.imshow(grady, cmap='gray')
ax2.set_title('Absolute grady Threshold bird eye', fontsize=40)


# In[19]:

mag_binary = mag_threshold(birdseye, sobel_kernel=9, mag_thresh = (30, 100))

plt.imshow(mag_binary, cmap='gray')
plt.title('Magnitude Threshold', fontsize=20)


# In[20]:

dir_binary = dir_threshold(birdseye, sobel_kernel= 15, thresh = (0.7, 1.3))
# plot the result
plt.imshow(dir_binary, cmap='gray')
plt.title('Directional Binary', fontsize=20)


# In[21]:

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
plt.imshow(combined, cmap='gray')


# In[22]:

def hls_select_s_channel(img):
    #convert to hls color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    return s_channel

#define a function that thresholds the S-channel of HSV
s_channel = hls_select_s_channel(birdseye)
# plot the result
f, (ax1, ax2) = plt.subplots(1,2, figsize=(24,9))
f.tight_layout()
ax1.imshow(cv2.cvtColor(birdseye, cv2.COLOR_BGR2RGB))
ax1.set_title('birdseye', fontsize = 30)
ax2.imshow(s_channel, cmap = "gray")
ax2.set_title('s_channel', fontsize=40)
   
    


# In[23]:

ksize = 15
sobelx_thresh = (20, 100)
sobely_thresh = (20, 100)
mag_grad_thresh = (20, 250)
dir_grad_thresh = (0.3, 1.3)

hls = cv2.cvtColor(birdseye, cv2.COLOR_RGB2HLS)

mask = np.zeros_like(hls[:,:,0])
mask[(hls[:,:,2]>200)& (hls[:,:,0]< 50)]=1
mag_binary = mag_threshold(hls, sobel_kernel=ksize, mag_thresh = mag_grad_thresh)
final_mask = np.maximum(s_channel, mag_binary)
plt.imshow(final_mask)




# gradx = abs_sobel_thresh(hls, orient='x', sobel_kernel=ksize, thresh=(20, 100))
# grady = abs_sobel_thresh(hls, orient='y', sobel_kernel=ksize, thresh=(20, 100))
# mag_binary = mag_threshold(hls, sobel_kernel=ksize, mag_thresh = mag_grad_thresh)
# dir_binary = dir_threshold(hls, sobel_kernel= 15, thresh = dir_grad_thresh)      
# binary_warped = np.zeros_like(hls)
# binary_warped[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
# plt.imshow(binary_warped)

# In[24]:

#fake data for default first previous lane
ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
# For each y position generate random x position within +/-50 pix
# of the line base position in each case (x=200 for left, and x=900 for right)
leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                              for y in ploty])
rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                for y in ploty])

leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


# Fit a second order polynomial to pixel positions in each fake lane line
left_fit = np.polyfit(ploty, leftx, 2)
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fit = np.polyfit(ploty, rightx, 2)
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


# In[25]:


histogram = np.sum(final_mask[final_mask.shape[0]//2:,:], axis=0)
plt.plot(histogram)


# In[26]:

# Take a histogram of the bottom half of the image
histogram = np.sum(final_mask[final_mask.shape[0]//2:,:], axis=0)
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
print('left max x value:', leftx_base,'right max x value:', rightx_base)


# In[27]:

# Find the offset of the car and the base of the lane lines
def find_offset(l_poly, r_poly):
    lane_width = 3.7  # metres
    h = 720  # height of image (index of image bottom)
    w = 1280 # width of image
    
    # Find the bottom pixel of the lane lines
    l_px = l_poly[0] * h ** 2 + l_poly[1] * h + l_poly[2]
    r_px = r_poly[0] * h ** 2 + r_poly[1] * h + r_poly[2]
    
    # Find the number of pixels per real metre
    scale = lane_width / np.abs(l_px - r_px)
    
    # Find the midpoint
    midpoint = np.mean([l_px, r_px])
    
    # Find the offset from the centre of the frame, and then multiply by scale
    offset = (w/2 - midpoint) * scale
    return offset


# In[34]:

# this variable is for the initial frame one 
# it allows the previous lines to be the above default lines
set_prev = 0
# Instanstiate lane lines for left and right
ci_left_line = Line()
ci_right_line = Line()



# In[35]:

def process_image(image):
    
    global left_fit, right_fit, set_prev, ci_right_line, ci_left_line
    
    if set_prev == 0:
        set_prev = 1
        right_fit_prev = right_fit
        left_fit_prev  = left_fit
 
    undist = corners_unwarp(image, mtx, dist)
   # define 4 source points for perspective transformation
    src = np.float32([[220,719],[1220,719],[750,480],[550,480]])
    # define 4 destination points for perspective transformation
    dst = np.float32([[240,719],[1040,719],[1040,0],[240,0]])
    Minv = cv2.getPerspectiveTransform(dst, src)
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp an image using the perspective transform, M:
    birdseye = cv2.warpPerspective(undist, M, (1280, 720), flags=cv2.INTER_LINEAR)
    birdseye = cv2.GaussianBlur(birdseye,(5,5),0)  
    hls = cv2.cvtColor(birdseye, cv2.COLOR_RGB2HLS)

    mask = np.zeros_like(hls[:,:,0])
    mask[(hls[:,:,2]>200)& (hls[:,:,0]< 50)]=1
    mag_binary = mag_threshold(hls, sobel_kernel=ksize, mag_thresh = mag_grad_thresh)
    binary_warped = np.maximum(mask, mag_binary)
    # Assuming you have created a warped binary image called "binary_warped"
# Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
# Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

# Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
# Choose the number of sliding windows
    nwindows = 9
# Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

# Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
 # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
# Set the width of the windows +/- margin
    margin = 100
# Set minimum number of pixels found to recenter window
    minpix = 50
# Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    #print('leftx_current',leftx_current)
# Step through the windows one by one
    for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
    # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
# Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
# Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
   
    # Fit a second order polynomial to each
    #left and right polyfits 
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
     #If no lines use previous   
    if (ci_left_line.line_detect(left_lane_inds)):
        left_fit = left_fit_prev
        
    if (ci_right_line.line_detect(right_lane_inds)):
        right_fit = right_fit_prev
       
         
    left_fitx = left_fit[0]*righty**2 + left_fit[1]*righty + left_fit[2]
    right_fitx = right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]
                                    
    left_fit_prev = left_fit
    right_fit_prev = right_fit                                
     # to cover same y-range as image in curvature
    ploty = np.linspace(0, 719, num=len(left_fitx))
    #left and right polyfits for curvature calculation
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    y_eval = 719
    
#define curvature 
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5)                              /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5)                                 /np.absolute(2*right_fit_cr[0])
        
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty ]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

  
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    undist = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    #middle = (lanes[0].allx[-1] + lanes[1].allx[-1])//2
    veh_pos = img.shape[1]//2
  
    #xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
    #off_center = (veh_pos - middle) * xm_per_pix # Positive if on right, Negative on left

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(undist, 'Radius of curvature (Left)  = %.2f m' % (left_curverad), (10, 40), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(undist, 'Radius of curvature (Right) = %.2f m' % (right_curverad), (10, 70), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(undist, 'Vehicle position : = %.2f m of center' % (find_offset(left_fitx, right_fitx)), 
               (10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    
    
    return undist


# In[36]:

def show_images(images, per_row=2, titles=None, main_title=None):
    figure = plt.figure(1)

    for n, img in enumerate(images):
        ax = figure.add_subplot(np.ceil(len(images) / per_row), per_row, n + 1)

        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        if (titles is not None and len(titles) >= n):
            ax.set_title(titles[n])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
        plt.imshow(img)
    if main_title is not None:
        plt.suptitle(main_title)

    plt.show()


# In[40]:

images_path = 'CarND-Advanced-Lane-Lines-master/SomeFrames/*.jpg'
image_files = glob.glob(images_path)

for fname in image_files:
     
    img = cv2.imread(fname)
    show_images([process_image(img)], titles=[fname])


# In[39]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
movie_output = 'movie.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(movie_output, audio=False)')


# xm_per_pix = 3.7/700 # meteres per pixel in x dimension
# screen_middel_pixel = img.shape[1]/2
# 
# left_lane_pixel = lane_info[6][0]    # x position for left lane
# right_lane_pixel = lane_info[5][0]   # x position for right lane
# car_middle_pixel = int((right_lane_pixel + left_lane_pixel)/2)
# screen_off_center = screen_middel_pixel-car_middle_pixel
# meters_off_center = xm_per_pix * screen_off_center

# Next Revision:  Make two copies of allX, and add -5 to one side and +5 to the other side. Do this for both lanes.  Make a polygon like you did with the road surface, but don't warp it back to perspective.  Just use it for masking for a faster video processing.  If your polynomials are good then they should be able to get you points that you can count for tracking confidence and then feed them to another polyfit to get you to your next polynomials
