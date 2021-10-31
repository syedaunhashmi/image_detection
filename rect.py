import cv2
from PIL import Image
from skimage.io import imshow
import matplotlib.pyplot as plt
#!/usr/bin/env python3
from scipy import ndimage
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
import numpy as np
import glob
import os
bufferP = []
from skimage.transform import rotate
from skimage.transform import (hough_line, hough_line_peaks)
from shapely.geometry import Polygon
from pprint import pprint as pp
from skimage.filters import threshold_otsu, sobel
window_name = 'crop'
from skimage import io
from scipy.stats import mode
from skimage.color import rgb2gray
debug_mode = False
def overlap2(rect1,rect2,org):
    p1 = Polygon([(rect1[0],rect1[1]), (rect1[2],rect1[1]),(rect1[2],rect1[3]),(rect1[0],rect1[3])])
    p2 = Polygon([(rect2[0],rect2[1]), (rect2[2],rect2[1]),(rect2[2],rect2[3]),(rect2[0],rect2[3])])

    # int_coords = lambda x: np.array(x).round().astype(np.int32)
    # exterior1 = [int_coords(p1.exterior.coords)]
    # exterior2 = [int_coords(p2.exterior.coords)]
    # overlay = org.copy()
    # cv2.fillPoly(overlay, exterior1, color=(255, 255, 0))
    # cv2.fillPoly(overlay, exterior2, color=(255, 0, 0))
    # cv2.addWeighted(overlay, 0.5, org, 1 - 0.5, 0, org)
    # cv2.namedWindow("Polygon", cv2.WINDOW_NORMAL)
    # cv2.imshow("Polygon", org)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return(p1.intersects(p2))


def rotate(image1):
    og = image1.copy()
    gray=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    dst = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 10)
    erode1 = cv2.erode(dst, None, iterations=1)

    edges = cv2.Canny(erode1,130, 255,apertureSize =3)

    canimg = cv2.Canny(gray, 1, 7)
    lines= cv2.HoughLines(canimg, 1, np.pi/180.0, 250, np.array([]))
    lines= cv2.HoughLines(edges, 1, np.pi/180, 80, np.array([]))
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(image1,(x1,y1),(x2,y2),(0,0,255),2)
    print(theta)
    print(rho)

    angle = 180*theta/3.1415926
    if (angle < 0):

        angle = angle + 90

    else:
        angle = angle - 90
    img_rotated = ndimage.rotate(og,angle,cval=255, reshape=True)
    # img_rotated = ndimage.rotate(img_rotated, 90, cval=255, reshape=True)
    cv2.imshow('asdaa',image1)
    cv2.waitKey(0)
    return img_rotated


def binarizeImage(RGB_image):
    image = rgb2gray(RGB_image)
    threshold = threshold_otsu(image)
    bina_image = image < threshold

    return bina_image


def findEdges(bina_image):
    image_edges = sobel(bina_image)
    return image_edges


def findTiltAngle(image_edges):
    h, theta, d = hough_line(image_edges)
    accum, angles, dists = hough_line_peaks(h, theta, d)
    angle = np.rad2deg(mode(angles)[0][0])

    if (angle < 0):

        r_angle = angle + 90

    else:

        r_angle = angle - 90

    # for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    #     y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    #     # ax.plot(origin, (y0, y1), '-r')
    return r_angle


def rotateImage(RGB_image, angle):
    fixed_image = rotate(RGB_image, angle)
    return fixed_image


def fix_crooked(img,c):
    org = img.copy()
    bina_image = binarizeImage(img)
    image_edges = findEdges(bina_image)
    angle = findTiltAngle(image_edges)
    average = sum(c) / len(c)
    img_rotated = ndimage.rotate(org, angle, cval=average, reshape=True)
    return img_rotated


def removeddf(img):
    ret, binary_img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv2.boundingRect(c)
        if int(rect[3]/rect[2]) < 3: continue
        x,y,w,h = rect
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Show",img)
        cv2.waitKey()
        cv2.destroyAllWindows()

def rmbg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 50,255, cv2.THRESH_BINARY_INV)
    cv2.imshow('ssdd',threshed)
    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    ## (4) Crop and save it
    x,y,w,h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]
    cv2.imshow('ss',img)
    cv2.imwrite("001.png", dst)
    cv2.waitKey(0)



def get_image_width_height(image):
    image_width = image.shape[1]  # current image's width
    image_height = image.shape[0]  # current image's height
    return image_width, image_height


def calculate_scaled_dimension(scale, image):
    # http://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
    image_width, image_height = get_image_width_height(image)
    ratio_of_new_with_to_old = scale / image_width
    dimension = (scale, int(image_height * ratio_of_new_with_to_old))
    return dimension


def rotate_image(image, degree=180):
    # http://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
    image_width, image_height = get_image_width_height(image)
    center = (image_width / 2, image_height / 2)
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    image_rotated = cv2.warpAffine(image, M, (image_width, image_height))
    return image_rotated


def scale_image(image, size):
    image_resized_scaled = cv2.resize(
        image,
        calculate_scaled_dimension(
            size,
            image
        ),
        interpolation=cv2.INTER_AREA
    )
    return image_resized_scaled

def detect_box(image,org, cropIt=True):
    # https://stackoverflow.com/questions/36982736/how-to-crop-biggest-rectangle-out-of-an-image/36988763
    # Transform colorspace to YUV
    image2 = org.copy()
    h0, w0, _ = image.shape
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_y = np.zeros(image_yuv.shape[0:2], np.uint8)
    image_y[:, :] = image_yuv[:, :, 0]

    global bufferP

    # if len(bufferP)>16:
    #     #     bufferP = bufferP[-10:]

    # Blur to filter high frequency noises
    image_blurred = cv2.GaussianBlur(image_y, (3, 3), 0)
    if (debug_mode):  show_image(image_blurred, window_name)

    # Apply canny edge-detector
    edges = cv2.Canny(image_blurred, 100, 200, apertureSize=5)
    if (debug_mode): show_image(edges, window_name)

    # Find extrem outer contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (debug_mode):
         #                                      b  g   r
         cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
         show_image(image, window_name)

    # https://stackoverflow.com/questions/37803903/opencv-and-python-for-auto-cropping
    # Remove large countours
    new_contours = []
    for c in contours:
        if cv2.contourArea(c) < 202033300:
            new_contours.append(c)

    # Get overall bounding box
    best_box = [-1, -1, -1, -1]
    for c in new_contours:
        x, y, w, h = cv2.boundingRect(c)
        if best_box[0] < 0:
            best_box = [x, y, x + w, y + h]
        else:
            if x < best_box[0]:
                best_box[0] = x -50
                if best_box[0] <0:
                    best_box[0]=0

            if y < best_box[1]:
                best_box[1] = y-50
                if best_box[1] < 0:
                    best_box[1]=0
            if x + w > best_box[2]:
                best_box[2] = x + w +50
                if best_box[2] > w0:
                    best_box[2] = w0
            if y + h > best_box[3]:
                best_box[3] = y + h+45
                if best_box[3] > h0:
                    best_box[3] = h0

    discard = False
    # if (best_box[0] + best_box[2]) / 2 > bufferP[i][0] and (best_box[1] + best_box[3]) / 2 > bufferP[i][1] and (best_box[0] + best_box[2]) / 2< bufferP[i][2] and (best_box[1] + best_box[3]) / 2< bufferP[i][3]:
    #     discard= not discard
    #     break
    rect1 = np.array([best_box[0]+80, best_box[1]+80, best_box[2]-100, best_box[3]-100])

    for b in bufferP:
        if (best_box[0] + best_box[2]) / 2 > b[0] and (best_box[1] + best_box[3]) / 2 > b[1] and (
                best_box[0] + best_box[2]) / 2 < b[2] and (best_box[1] + best_box[3]) / 2 < b[3]:
            print("discarded by roi")
            discard = not discard
            break
        rect2 = np.array ([b[0]+90, b[1]+90, b[2]-90, b[3]-90])
        # cv2.rectangle(org, (best_box[0], best_box[1]) ,(best_box[2], best_box[3]), (0, 255, 0), 3)
        # cv2.rectangle(org, (b[0]+30, b[1]+30), (b[2]-140, b[3]-140), (0, 0, 255), 3)
        # cv2.namedWindow("rect compare", cv2.WINDOW_NORMAL)
        # cv2.imshow('rect compare',org)
        if overlap2(rect1,rect2,org):
            discard= not discard
            break





    if (debug_mode):
        cv2.rectangle(image, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 0, 0), 4)
        show_image(image, 'window_name')
        print(best_box)

    if (cropIt):
        image2 = image2[best_box[1]:best_box[3], best_box[0]:best_box[2]]
        if (debug_mode): show_image(image2, window_name)

    if discard :
        print('discarded')
        return None,None
    else:
        bufferP.append(best_box)
        return image2,best_box


def show_image(image, window_name):
    # Show image
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    image_width, image_height = get_image_width_height(image)
    cv2.resizeWindow(window_name, image_width, image_height)

    # Wait before closing
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cut_of_top(image, pixel):
    image_width, image_height = get_image_width_height(image)

    # startY, endY, startX, endX coordinates
    new_y = 0+pixel
    image = image[new_y:image_height, 0:image_width]
    return image

def cut_of_bottom(image, pixel):
    image_width, image_height = get_image_width_height(image)

    # startY, endY, startX, endX coordinates
    new_height = image_height-pixel
    image = image[0:new_height, 0:image_width]
    return image





def crop_image_v2(img):
    mask = img!=255
    mask = mask.any(2)
    mask0,mask1 = mask.any(0),mask.any(1)
    colstart, colend = mask0.argmax(), len(mask0)-mask0[::-1].argmax()+1
    rowstart, rowend = mask1.argmax(), len(mask1)-mask1[::-1].argmax()+1
    return img[rowstart:rowend, colstart:colend]




from matplotlib import cm

def back(j,img,single=False):
    #== Parameters =======================================================================
    BLUR = 15
    CANNY_THRESH_1 = 30
    CANNY_THRESH_2 = 250
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (1.0,1.0,1.0) # In BGR format


    #== Processing =======================================================================

    #-- Read image -----------------------------------------------------------------------
    if not single:
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        # im = Image.open(img)
        pix = im.load()
    else:
        img = imgS
    colorOfImg =pix[2, 2]
    print('image:',j,' value=',colorOfImg)
    org = img.copy()
    if all(i >= 235 for i in colorOfImg):
        print('white')
        BLUR = 15
        CANNY_THRESH_1 = 5
        CANNY_THRESH_2 = 10
        MASK_DILATE_ITER = 10
        MASK_ERODE_ITER = 10
        org = img.copy()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        img_thresh = img
        thresh = 180
        img_thresh[img_thresh >= thresh] = 255

    elif all( 235 >i >= 176 for i in colorOfImg):
        BLUR = 15
        CANNY_THRESH_1 = 20
        CANNY_THRESH_2 = 255
        MASK_DILATE_ITER = 10
        MASK_ERODE_ITER = 10
        CANNY_THRESH_1 = 10
        CANNY_THRESH_2 =90
        print('color resgment')
        final = img
        # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # cl = clahe.apply(l)
        # limg = cv2.merge((cl, a, b))
        # final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # img_thresh = img
        # thresh = 180
        # img_thresh[img_thresh >= thresh] = 255

    else:
        CANNY_THRESH_1 = 10
        CANNY_THRESH_2 = 90
        final = img
        # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # cl = clahe.apply(l)
        # limg = cv2.merge((cl, a, b))
        # final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # img_thresh = img
        # thresh = 180
        # img_thresh[img_thresh >= thresh] = 255
    gray = cv2.cvtColor(final,cv2.COLOR_BGR2GRAY)
    height, width,_= img.shape

    # area is calculated as “height x width”
    area = (height * width)
    print('areaimg,',area)
    #-- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Previously, for a previous version of cv2, this line was:
    #  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Thanks to notes from commenters, I've updated the code but left this note
    for c in contours:
        # if cv2.contourArea(c) < area*0.05:
        #     continue
        # if cv2.contourArea(c)> area*0.94:
        #     continue
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))

    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

    # contour_info = contour_info[:-3]
    # try:
    #      max_contour = contour_info[i]
    # except:
    #     return 0

    pathlist=[]
    looper =1
    for i in range(0,10):
        if cv2.contourArea(contour_info[i][0]) < area*0.001:
            continue
        # if len(contour_info)< 4000:
        #     cv2.imwrite('cropped/{}({}).jpg'.format(j, i), org)
        #     return 0

        #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
        # Mask is black, polygon is white
        mask = np.zeros(edges.shape)
        cv2.fillConvexPoly(mask, contour_info[i][0], (255))

    #-- Smooth mask, then blur it --------------------------------------------------------
        mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
        mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
        mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
        mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

        #-- Blend masked img into MASK_COLOR background --------------------------------------
        mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices,
        img         = img.astype('float32') / 255.0                 #  for easy blending

        masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
        masked = (masked * 255).astype('uint8')     # Convert back to 8-bit
        ## corners detection
        newimg = masked.copy()


        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow('img', newimg)  # Display
        # cv2.waitKey()
        image1 , p = detect_box(newimg,org, True)
        if image1 is None:
            print("no proper image")
            continue
        if not single:
            # image1 = rotate(image1)
            image1 = fix_crooked(image1,colorOfImg)
            image1 = fix_crooked(image1,colorOfImg)
            h,w,channels  =image1.shape
            image1 = image1[85:h-90, 90:w-85]

            try:
                pathlist.append('cropped/{}({}).jpg'.format(j,looper))
                cv2.imwrite('cropped/{}({}).jpg'.format(j,looper),image1)
                looper+=1
            except:
                continue
        else:
            return image1
        i+=1
    return pathlist


import os, shutil
def delete_work(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))



def process_image(path):
    img = cv2.imread('images_to_work/{}'.format(path))
    np=os.path.splitext(path)[0]
    paths=back(np, img)
    bufferP.clear()
    delete_work('images_to_work')
    return paths
