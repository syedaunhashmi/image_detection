import numpy as np
import cv2
from scipy import ndimage

def rotate(image1):
    og = image1.copy()
    gray=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,50,150,apertureSize = 6)

    canimg = cv2.Canny(gray, 20, 250)
    lines= cv2.HoughLines(canimg, 1, np.pi/180.0, 250, np.array([]))
    # lines= cv2.HoughLines(edges, 1, np.pi/180, 80, np.array([]))
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
    img_rotated = ndimage.rotate(og,angle,cval=255, reshape=True)
    return img_rotated
    # cv2.imshow('asdaa',img_rotated)
    # cv2.waitKey(0)


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


img = cv2.imread('2.jpg')
img = rotate(img)
removeddf(img)
