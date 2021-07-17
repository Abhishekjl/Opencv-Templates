# import numpy as np
# import cv2



# img = cv2.imread('samples/lena.jpg',flags = 1)

# # print(img.shape)
# cv2.imshow('wn', img)
# cv2.waitKey(0)


# working on camera 
# import cv2
# import numpy as np

# cap = cv2.VideoCapture(0) # device index 
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
   
# size = (frame_width, frame_height)
   
# fourcc = cv2.VideoWriter_fourcc(*'MJPG') # video codec 
# output = cv2.VideoWriter('myvideo.avi', 
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          20, size)

# # while(True):
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # this shows width and height
#         # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # ret is a boolean value here
#         # frame = cv2.resize(frame,(1210,720))
#         cv2.imshow('frame',frame)
#         output.write(frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     else:
#         break    
# cap.release()
# output.release()
# cv2.destroyAllWindows()

# working with shapes in opecv


# import cv2
# import numpy as np

# img = np.ones(shape = (640,640,3))
# img = cv2.line(img, pt1 = (0,0),pt2 = (640,640), color = (0,0,255), thickness = 3)
# img = cv2.arrowedLine(img, pt1 = (0,640), pt2 = (320,320), color = (0,255,0), thickness = 3)
# img = cv2.rectangle(img, pt1 = (160,160), pt2 =(480,480), color = (255,0,0), thickness=4)
# img = cv2.circle(img, center = (320,320), radius=160, color=(0,0,255), thickness=4)
# img = cv2.putText(img, text = 'opencv', org = (200,50), fontFace =cv2.FONT_HERSHEY_SIMPLEX,fontScale = 2,thickness=4, color= (0,0,255),lineType=cv2.LINE_AA )


# cv2.imshow('window', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# parameters in capture and images and put date and time
# import cv2
# import numpy as np
# import datetime
# cap = cv2.VideoCapture(0)

# # cap.set(3,300.0) # 3 means width 
# # cap.set(4,300.0) # 4 means height
# # print(cap.get(3), cap.get(4))


# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         text = 'Width: ' + str(cap.get(3)) + 'Height: ' + str(cap.get(4))
#         date = str(datetime.datetime.now())
#         frame = cv2.putText(frame, text = date, org = (10,50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, thickness=1, fontScale=1, lineType=cv2.LINE_AA, color=(0,0,255) )

#         cv2.imshow('window', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()        


# handling the mouse events 
# import cv2 
# import numpy as np
# events  = [i for i in dir(cv2) if 'EVENT' in i]
# # print(events)

# def click_event(event,x,y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, y)
#         font = cv2.FONT_HERSHEY_COMPLEX
#         text = str(x) + ',' + str(y)
#         cv2.putText(img, text, (x,y), font, 1, (255,255,0), 2, cv2.LINE_AA )
#         cv2.imshow('window',img)
    
#     if event == cv2.EVENT_RBUTTONDOWN:
#         blue = img[y,x,0]
#         green = img[y,x,1]
#         red = img[y,x,2]
#         print(blue, red, green)
#         text_channel = str(blue)+' '+str(green) + ' ' + str(red)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(img, text = text_channel, org = (x,y), fontFace = font, fontScale = 1, color = (0, 255, 255), thickness = 1)
#         cv2.imshow('window', img)


# img = cv2.imread('samples/lena.jpg')        
# # img = np.ones((512,512,3), np.uint8)
# cv2.imshow('window', img)
# cv2.setMouseCallback('window', click_event)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# --------------------------------------------------------------------------more on events 
# import cv2
# import numpy as np


# def click_event(event, x,y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x,y)
#         points.append((x,y))
#         if len(points)>=2:
#             cv2.line(img, points[-1], points[-2], color = (255,0,0), thickness=2, lineType=cv2.LINE_AA)

#         cv2.circle(img, (x,y), radius = 5, color = (0,0,255), thickness=-1)
#         cv2.imshow('window', img)

# img = np.ones((512,512,3))
# cv2.imshow('window', img)
# points = []
# cv2.setMouseCallback('window', click_event)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# color picker using mouse event 
# import cv2
# import numpy as np


# def click_event(event, x,y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x,y)
#         blue = img[x,y,0]
#         green = img[x,y,1]
#         red = img[x,y,2]
#         cv2.circle(img, (x,y), 3, (0,0,255), -1)
#         myCOLORimg = np.zeros((512,512,3), np.uint8)
#         myCOLORimg[:] = [blue, green, red]
#         cv2.imshow('window1', myCOLORimg)

        

# img = cv2.imread('samples/lena.jpg')
# img = cv2.resize(img, (800,800))
# cv2.imshow('window', img)
# cv2.setMouseCallback('window', click_event)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


####----------------------------------------------------------------- bullseye---------------------##

# import cv2
# import numpy as np
# # drawing figure using opencv
# canvas = np.ones((600,600,3))
# (center_x, center_y) = canvas.shape[0]//2, canvas.shape[1]//2
# for i in range(0,300, 5):
#     cv2.circle(canvas, center = (center_x,center_y), radius = i , thickness=1, color = (0,0,255))

# cv2.imshow('window', canvas)
# cv2.waitKey(0)


#### ==================### 
# ----------------------------------------------------------------------making a new random picture ###
# import cv2
# import numpy as np

# canvas = np.ones((700,1000,3), np.uint8)
# for i in range(0,100):
#     radius = np.random.randint(5, high = 500)
#     color = np.random.randint(0, high = 255, size = (3,)).tolist()
#     # print(tuple(color))
#     pt = np.random.randint(0, high = 700, size = (2,))
#     cv2.circle(canvas, tuple(pt), radius, tuple(color), -1)
# cv2.imshow('window', canvas)
# cv2.waitKey(0)


# import numpy as np
# import cv2

# img = cv2.imread('samples/lena.jpg')
# b,g,r = cv2.split(img)
# print(b.shape, g.shape, r.shape)
# img_merged = cv2.merge((b,g,r))
# print('merged', img_merged.shape)
# print(img_merged.dtype)


# cv2.imshow('windows', img_merged)
# cv2.waitKey(0)


# -------------------------------------------------------image cropping by using the mouse callbacks 

# import cv2
# import numpy as np

# flag = False
# ix = -1
# iy = -1

# def mouse_event(event, x,y, flags, params):
#     global flag, ix, iy
    
#     if event == cv2.EVENT_RBUTTONDOWN:
#         flag = True
#         ix = x
#         iy = y
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if flag == True:
#             clone = img.copy()

#             cv2.rectangle(clone, pt1 = (ix,iy), pt2 =(x,y), color = (255,255,255), thickness = 3)
#             cv2.imshow("window", clone)
    
#     elif event == cv2.EVENT_RBUTTONUP:
#         fx = x
#         fy = y
       
#         flag = False

#         cv2.rectangle(img, pt1 = (ix,iy), pt2 = (x,y), color = (255,255,255), thickness  = 3)
#         cropped = img.copy()[iy:fy, ix:fx]
#         cv2.imshow('cropped', cropped)
#         cv2.imwrite('cropped.png', cropped)
# cv2.namedWindow(winname = 'window')
# img = cv2.imread('samples/lena.jpg')
# cv2.setMouseCallback('window', mouse_event)


# while True:
#     cv2.imshow('window', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break  
# cv2.destroyAllWindows()                                      

# # img2 = np.ones((600,600,3))
# canvas = np.ones((400,800,3))
# cv2.rectangle(canvas, (400,0), (800,400), thickness=-1, color = (0,0,0))
# cv2.rectangle(canvas, (0,0), (400,400), thickness=-1, color = (255,255,255))

# cv2.imshow('wnd', canvas)
# cv2.imwrite('bit.jpg', canvas)
# cv2.waitKey(0)

# ------------------------------------------------------------arthemetic operation in images addWeighted()
# import cv2
# import numpy as np
# img1 = cv2.imread('samples/lena.jpg')
# img2 = cv2.imread('samples/aloeL.jpg')
# img1 = cv2.resize(img1, (600,600))
# img2 = cv2.resize(img2, (600,600))
# dst = cv2.addWeighted(img1,.6,img2,.3,0) # it changes to watermark
# cv2.imshow('window', dst)
# cv2.waitKey(0)



#####------------------------------------------------- bitwise operator bitwise_and, bitwise_or,bitwise_xor
# import cv2
# import numpy as np
# img1 = np.zeros((400,800,3), np.uint8)
# img1 = cv2.rectangle(img1, (200,0), (700,100), (255,255,255), -1)
# img2 = cv2.imread('bit.jpg')
# print(img1.shape, img2.shape)
# bitAnd = cv2.bitwise_or(img2, img1)
# # print(bitAnd.shape)
# cv2.imshow('window', img2)
# cv2.imshow('window2', img1)
# cv2.imshow('window3', bitAnd)

# cv2.waitKey(0)


###---------------------------------------------------------------------------------- making trackbar 
# import numpy as np
# import cv2

# def nothing(x):
#     print(x)


# img = np.zeros((300,512,3), np.uint8)
# cv2.namedWindow('image')
# cv2.createTrackbar('Blue', 'image',0,255, nothing )
# cv2.createTrackbar('Green', 'image',0,255, nothing )
# cv2.createTrackbar('Red', 'image',0,255, nothing )
# # switch
# switch = '0 : OFF\n 1: ON'
# cv2.createTrackbar(switch, 'image', 0,1,nothing)


# while True:
#     cv2.imshow('image', img)
    

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break  

#     blue =cv2.getTrackbarPos('Blue','image')
#     green =cv2.getTrackbarPos('Green','image')
#     red =cv2.getTrackbarPos('Red','image')
#     s = cv2.getTrackbarPos(switch, 'image')
#     if s == 0:
#         img[:] = 0
#     if s == 1:    
#         img[:] = [blue, green , red]


# cv2.destroyAllWindows()



# object detection using color hsv, inRange, masking
# 


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# img = cv2.imread('samples/sudoku.png',0)

# #----------------------------------------------------------------------- image thresholding 

# _, th1 = cv2.threshold(img, thresh = 124, maxval = 255,type=cv2.THRESH_BINARY)
# _, th2 = cv2.threshold(img, thresh = 124, maxval = 255,type=cv2.THRESH_BINARY_INV)
# _, th3 = cv2.threshold(img, thresh = 140, maxval = 255,type=cv2.THRESH_TRUNC)
# # thresh_trunk makes a line by changed threshold
# _, th4 = cv2.threshold(img, thresh = 124, maxval = 255,type=cv2.THRESH_TOZERO)
# thresh_adaptive = cv2.adaptiveThreshold(img, 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C,
#                                 thresholdType = cv2.THRESH_BINARY, blockSize = 11, C = 5)
# _, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# thresh_adaptive_gaussing = cv2.adaptiveThreshold(img, 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                          thresholdType = cv2.THRESH_BINARY, blockSize = 11, C = 5)

# # here C removes the noise from the image
# titles = ['Original_image', 'BIANRY','THRESH_BINARY_INV','THRESH_TRUNC','THRESH_TOZERO','ADAPTIVE_THRESH_MEAN','THRESH_GAUSSSIAN']
# images = [img, th1, th2, th3, th4,thresh_adaptive, thresh_adaptive_gaussing ]
# for i in range(len(titles)):
#     plt.subplot(2,4,i+1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
# plt.show()

# #  
# cv2.imshow('window',img)
# cv2.imshow('thresholding',  thresh_adaptive)
# cv2.imshow('thresholding1',  th2)
# cv2.imshow('tresh_gaussian',  thresh_adaptive_gaussing)
# # cv2.imshow('thresholding3',  th4)



# cv2.waitKey(0)


# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# img = cv2.imread('samples/lena.jpg')

# cv2.imshow('image', img)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(img.shape)
# plt.imshow(img)
# # plt.xticks([])
# # plt.yticks([]) #manual ticks of the axis
# plt.show()
# cv2.waitKey(0)


# -----------------------------------morphological transformation using opencv

# its some simple operation based on the shape of image, generally applied on binary image
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('j1.jpg', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img, thresh=220, maxval= 255, type = cv2.THRESH_BINARY_INV)
kernel = np.ones((2,2)) # bigger kernel size means more merged picture 
dilation = cv2.dilate(mask,kernel, iterations=3)
erosion = cv2.erode(dilation, kernel,iterations = 4) # making boundary clean 
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 3) # first erosion and then dilation = opening morhpology
closing =cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 3) # first dilation  then erosion
morph_gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel, iterations = 3)
morph_tophat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel, iterations = 3)


titles = ['img','mask','dilation','erosion','opening', 'closing', 'morph_gradient','morph_tophat']

images = [img, mask, dilation, erosion, opening, closing, morph_gradient, morph_tophat]

for i in range(len(titles)):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
plt.show()    
#########---------------------------------Smoothing and blurring in images --------## to remove noise 
# here we use linear filter mostly 
# kernal is a numpy imge for convolving on the image0
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# img = cv2.imread('samples/lena.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# kernel  = np.ones((5,5), np.float32)/(5*5)

# dst = cv2.filter2D(img, ddepth = -1, kernel = kernel) # this smoothen and removed noises 
# # AS in 1-D signals images also can be filtered by using low-pass-filter(removing noise, blur), high-pass-filter(fiding-edge)
# blur = cv2.blur(img, (5,5))

# # gaussian filter kernel is nothing but using different-weight-kernel, in both x and y direction
# gaussian_blur = cv2.GaussianBlur(img, (5,5),0) # gaussian blur is better to remove high frequency noise

# # median filter ( great when dealing with salt and pepper noise)(white and black pixels noise )
# median_filter  = cv2.medianBlur(img, 5)  # kernel size must be odd


# # bilateral_filter = > blur it by preserving its edges
# bilateral_filter = cv2.bilateralFilter(img, 9,75,75) # its highly effective while keeping border sharp



# titles = ['original','2dConv', 'blur','gaussian_blur','median_filter','bilateral_filter']
# images = [img,dst, blur, gaussian_blur, median_filter, bilateral_filter]

# for i in range(len(titles)):
#     plt.subplot(2,3,i+1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
# plt.show()    

# --------------------------------------------image gradient and canny_edge detection

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# img = cv2.imread('samples/sudoku.png', cv2.IMREAD_GRAYSCALE)
# #laplacian gradient

# laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)    #cv2.CV64F is a data type
# laplacian = np.uint8(np.absolute(laplacian))

# # -------------------------------------------------------sobel gradient
# sobelX = cv2.Sobel(img, cv2.CV_64F, dx = 1 , dy = 0)
# sobelY = cv2.Sobel(img, cv2.CV_64F, dx = 0, dy = 1)
# # we can also replace cv2.CV_64F with -1 and then we dont need to use np.uint8(np.absolute())
# sobelX = np.uint8(np.absolute(sobelX))
# sobelY = np.uint8(np.absolute(sobelY))
# sobelXY =  cv2.bitwise_or(sobelX,sobelY)
# sobelXY_add = cv2.addWeighted(sobelX, 1, sobelY , 1, 0)
# titles = ['image','laplacian','sobelX', 'sobelY','sobel_combined','sobel_added']
# images = [img, laplacian, sobelX, sobelY, sobelXY, sobelXY_add]

# for i in range(len(titles)):
#     plt.subplot(3,2, i+1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
# plt.show()


#----------------------------------------------- canny edge detector 

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# img = cv2.imread('samples/messi5.jpg',0)
# canny = cv2.Canny(img, threshold1 = 100, threshold2 = 200)
# laplacian = cv2.Laplacian(img, -1,3)

# titles = ['source', 'canny','laplacian']
# images = [img, canny, laplacian]

# for i in range(len(titles)):
#     plt.subplot(2,2, i+1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
# plt.show()

# -------------------------------------------------------------------------------------Image pyramids 
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# img = cv2.imread('samples/lena.jpg')
# pyr_down = cv2.pyrDown(img) # here we loose information of image while reducing resolution
# pyr_up = cv2.pyrUp(pyr_down) 
# cv2.imshow('original_img', img)
# cv2.imshow('pyr_down', pyr_down)
# cv2.imshow('pyr_up', pyr_up)

# cv2.waitKey(0)

### -----------------------------------------------------------image blending -----------------------------
# import cv2
# import numpy as np
# import matplotlib.pyplot as  plt
# apple = cv2.imread('samples/apple.jpg')
# orange = cv2.imread('samples/orange.jpg')
# print(apple.shape, orange.shape)

# apple_orange = np.hstack((apple[:,:256], orange[:,256:]))

# #generate gaussian pyramid
# apple_copy = apple.copy()
# gp_apple = [apple_copy]
# for i in range(6):
#     apple_copy = cv2.pyrDown(apple_copy)
#     gp_apple.append(apple_copy)

# orange_copy = orange.copy()
# gp_orange = [orange_copy]
# for i in range(6):
#     orange_copy = cv2.pyrDown(orange_copy)
#     gp_orange.append(orange_copy)

# # generate laplacian pyramid
# apple_copy = gp_apple[5]
# lp_apple = [apple_copy]
# for i in range(5,0,-1):
#     gaussian_expanded = cv2.pyrUp(gp_apple[i])
#     laplacian = cv2.subtract(gp_apple[i-1], gaussian_expanded)
#     lp_apple.append(laplacian)    


# orange_copy = gp_orange[5]
# lp_orange = [orange_copy]
# for i in range(5,0,-1):
#     gaussian_expanded = cv2.pyrUp(gp_orange[i])
#     laplacian = cv2.subtract(gp_orange[i-1], gaussian_expanded)
#     lp_orange.append(laplacian)    


# # now add left and the right part 
# apple_orange_pyramid = []
# n = 0 
# for apple_lap, orange_lap in zip(lp_apple, lp_orange):
#     n += 1
#     cols, rows, ch = apple_lap.shape
#     laplacian = np.hstack((apple_lap[:,0:int(cols/2)], orange_lap[:,int(cols/2):]))
#     apple_orange_pyramid.append(laplacian)

# # now reconstruct the whole image
# apple_orange_reconstruct = apple_orange_pyramid[0]
# for i in range(1, 6):
#     apple_orange_reconstruct  = cv2.pyrUp(apple_orange_reconstruct)
#     apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i], apple_orange_reconstruct)




# cv2.imshow('apple', apple)
# cv2.imshow('orange', orange)
# cv2.imshow('added', apple_orange_reconstruct)

# cv2.waitKey(0)




########---------------------------------Countours finding-------------------------------------------
#countours are the curve joining all the continous points along with the boundary which have same color and boundary
# for better accuracy we use binary image nd also use edge detector with contours

# import numpy as np
# import matplotlib.pyplot as plt
# import cv2

# img = cv2.imread('samples/HappyFish.jpg')
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
# contours, hierarchy = cv2.findContours(thresh,mode =  cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
# # here contours is the list of coordinates(X,Y), hierarchy (contains information of image topology)
# print('number of contours = ' + str(len(contours)))

# ### drawing the contours
# cv2.drawContours(img,contours, -1,(0,255,0), 2)

# cv2.imshow('source', img)
# cv2.imshow('img_gray', img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# --------------------------------motion detection using opencv-------------------------------
# import cv2
# import numpy as np
# cap  = cv2.VideoCapture('samples/vtest.avi')
# ret, frame1 = cap.read()
# ret, frame2 = cap.read()

# # def nothing(x):
# #     pass

# # cv2.namedWindow('trackbar')

# # cv2.createTrackbar('thresh', 'trackbar',0,255, nothing)

# while cap.isOpened():
#     # ret, frame = cap.read()
#     diff = cv2.absdiff(frame1, frame2)  # this method is used to find the difference bw two  frames
#     gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0 )
#     # thresh_value = cv2.getTrackbarPos('thresh', 'trackbar')
#     _, threshold = cv2.threshold(blur, 23, 255, cv2.THRESH_BINARY)
#     dilated = cv2.dilate(threshold, (1,1), iterations=1)
#     contours, _, = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

#     # DRAWING RECTANGLE BOXED
#     for contour in contours:
#         (x,y,w,h) = cv2.boundingRect(contour)
#         if cv2.contourArea(contour) <400:
#             continue
#         cv2.rectangle(frame1, (x,y),(x+w, y+h), (0,255,0), 2)
#         cv2.putText(frame1, 'status: movement',(10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    
#     # cv2.drawContours(frame1,contours, -1, (0,255,0), 2)
#     cv2.imshow('frame',frame1)
#     frame1 = frame2
#     ret, frame2 = cap.read()

#     # cv2.imshow('inter',dilated)
#     # cv2.imshow('blur', blur)
#     # cv2.imshow('threshold', threshold)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#     # if cv2.waitKey(40) == 27:
#         break
# cv2.destroyAllWindows()


## ---------------------------Shape detection using opencv--------------------------------
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# img = cv2.imread('shapes.jpg')
# img_gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(img_gray, 240,255,cv2.THRESH_BINARY)
# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# for contour in contours:
#     approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour, closed = True),closed = True) # epsilon is polygon approximation accuracy
#     # this approx is the list of coordinates of shapes 
#     cv2.drawContours(img,contours = [approx],contourIdx = 0, color = (0,255,0), thickness = 5)
#     x = approx.ravel()[0]
#     y = approx.ravel()[1]
#     # this flattens the approx coordinates 
#     if len(approx) == 3:
#         cv2.putText(img, 'triangle', (x,y),  cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,0), 1)
#     if len(approx) == 4:
#         x,y,  w, h = cv2.boundingRect(approx)
#         aspectRatio = float(w/h)
#         if aspectRatio >= 0.95 and aspectRatio <=1.05:
#             cv2.putText(img, 'square', (x,y),  cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,0), 1)
#         else:
#             cv2.putText(img, 'rectangle', (x,y),  cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,0), 1)



#     elif len(approx) == 5:
#         cv2.putText(img, 'pentagon', (x,y),  cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,0), 1)  
#     elif len(approx) == 10:
#         cv2.putText(img, 'star', (x,y),  cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,0), 1)        
#     else:
#         cv2.putText(img, 'shape', (x,y),  cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,0), 1)   


# plt.imshow(img, 'gray')
# plt.show()
# cv2.waitKey(0)


#-------------------------------------------- understanding the image histogram ----------------------------
# this is the way to understand about contrast, brightness , pixel intensity
# import numpy as np
# import  cv2
# import matplotlib.pyplot as plt
# img = cv2.imread('samples/butterfly.jpg',0) # histogram for grayscale images
# print(img.shape)
# cv2.imshow('window', img)
# cv2.waitKey(0)
# img = np.zeros((200,200),  np.uint8)
# cv2.rectangle(img, (0,100), (200,200), color = (255,255,255), thickness = -1)
# cv2.rectangle(img, (0,50), (100,100), color = (125,125,125), thickness = -1)

# b,g,r = cv2.split(img)
# cv2.imshow('img', img)
# cv2.imshow('blue',b)
# cv2.imshow('green',g)
# cv2.imshow('red',r)
# plt.hist(b.ravel(),bins = 256, range=[0,256])
# plt.hist(g.ravel(),bins = 256, range=[0,256])
# plt.hist(r.ravel(),bins = 256, range=[0,256])

# # here we can also use calcHist method
# equalised = cv2.equalizeHist(img)
# hist = cv2.calcHist([img], channels = [0], mask = None, histSize = [256], ranges = [0,256])
# equalised_hist = cv2.calcHist([equalised], channels = [0], mask = None, histSize = [256], ranges = [0,256])

# plt.plot(hist)
# plt.plot(equalised_hist)
# cv2.imshow('window', img)
# cv2.imshow('window2', equalised)
# plt.show()
# cv2.waitKey(0)


# -----------------histogram equalisation improves the illumination and image quality----------------------

# ---------------------------------------------hough transformation-----------------
# the hough transformation is a popular technique to detect any shape if that shape can 
# represent into a mathematical form, even if the shape is broken or distorted it can be detect

# in opencv there are two types of hough transformation (standard and probabilistic)
# and this take 4 steps 


# import cv2
# import numpy as np
# img = cv2.imread('samples/sudoku.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# lines = cv2.HoughLines(edges, 1,np.pi/180, 200) # here 200 is the voting parameter 
# # these lines are in polar coordinates
# # print(len(lines))
# # hee lines will return the coordinates edge points

# for line in lines:
#     line = line[0]#theta is angle, and rho is the intercept 
#     rho = line[0]
#     theta = line[1]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     xO = a*rho # origin or top left corner of image
#     yO = b*rho
#     # X1 stores the rounded off value of (r*cos(theta) - 1000*sin(theta))
#     x1 = int(xO +1000*(-b))
#     # y1 stores the rounded value of (r*sin(theta) + 1000*(cos(theta)))
#     y1 = int(yO +1000*(a))

#     x2 = int(xO - 1000*(-b))
#     # x2 stores the rounded off value of (r*cos(theta) + 1000*sing(theta))
#     y2 = int(yO -1000*(a))
#     cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

# cv2.imshow('image', img)
# cv2.imshow('edges', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()    



# -------------------------------------probability hough transformation-------------------------------
# this is optimation of standard hough transformation

# import cv2
# import numpy as np
# img = cv2.imread('samples/sudoku.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# cv2.imshow('edges', edges)
# lines = cv2.HoughLinesP(edges, rho=1, theta = np.pi/180,threshold = 100, minLineLength = 100, maxLineGap = 10)
# # threshold means voting for the lines
# for line in lines:
#     x1,y1, x2,y2 = line[0]
#     cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)

# cv2.imshow('image', img)
# cv2.waitKey(0)


### -----------------------------------------hough circle transformation

# import numpy as np
# import cv2
# img = cv2.imread('samples/smarties.png')
# output = img.copy()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_blur = cv2.medianBlur(gray, ksize = 5)


# circles =cv2.HoughCircles(gray_blur, method = cv2.HOUGH_GRADIENT, dp = 1, minDist = 20,param1 = 50, param2 = 30,
#      minRadius=0, maxRadius= 0  )


# detected_circles = np.uint16(np.around(circles))
# # print(detected_circles)
# for (x,y,r) in detected_circles[0,:]:
#     cv2.circle(output, center = (x,y), radius= r, color = (0,255,0), thickness=3)
#     cv2.circle(output, center = (x,y), radius= 2, color = (255,0,0), thickness=3)

# cv2.imshow('output', output)
# cv2.waitKey(0)



 ### ---------------------object detection using haar cascade 
# import cv2
# face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_dectector = cv2.CascadeClassifier('haarcascade_eye.xml')

# # reading the input image now

# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(gray,1.1, 4 )
#     for (x,y, w, h) in faces:
#         cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  3)
#         roi_gray = gray[y:y+h,x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
#         eyes = eye_dectector.detectMultiScale(roi_gray)
#         for (ex,ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 5)

#     cv2.imshow("window", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# frame.release()


# img = cv2.imread('samples/basketball1.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_detector.detectMultiScale(gray,1.1, 4 )

# for (x,y, w, h) in faces:
#     cv2.rectangle(img, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  3)


# cv2.imshow("window", img)
# cv2.waitKey(0)





##################################
# import cv2

# cap = cv2.VideoCapture(0)

# while True:
#     _, frame  = cap.read()
#     cv2.imshow('window', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()

####  --------------------------------- background substraction method---------------------------------------
# this is for object tracking 

# import cv2
# import numpy as np
# from cv2 import bgsegm
# cap = cv2.VideoCapture('samples/vtest.avi')

# # fgbg = bgsegm.createBackgroundSubtractorMOG()
# # fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# fgbg = cv2.createBackgroundSubtractorKNN()


# while True:
#     ret, frame = cap.read()
#     if  frame is None:
#         break
#     foreground_mask  = fgbg.apply(frame)
#     cv2.imshow('window', frame)
#     cv2.imshow('masked', foreground_mask)



#     keyboard = cv2.waitKey(27)
#     if keyboard == 'q' or keyboard == 27:
#         break

#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break

# cap.release()    