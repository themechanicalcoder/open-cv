import cv2
import imutils
import numpy as np
from collections import deque

pts=deque(maxlen=(64))
for i in range(20):
    pts.appendleft(None)
cap=cv2.VideoCapture(0)
counter=0

#defining the colour range
lower=(10,50,50)
upper=(70,255,255)

while(True):
    ret,image=cap.read()
    image=imutils.resize(image,width=1000)

    #removing unnecessary noise from the captured image
    blurred=cv2.GaussianBlur(image,(11,11),0)
    hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

    #checking if the colour is in range
    mask=cv2.inRange(hsv,lower,upper)

    #dilating and eroding to remove the remaning blobs
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)
    #cv2.imshow("yellow",mask)
    #finding the contours

    cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=cnts[0] if imutils.is_cv2() else cnts[1]
    center=None

    #if an contour is found in the mask

    if(len(cnts)>0):

        #finding the contour with maximum area

        c=max(cnts,key=cv2.contourArea)
        ((x,y),radius)=cv2.minEnclosingCircle(c)

        #finding the centroid
        M=cv2.moments(c)
        try:
            center=(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
        except ZeroDivisionError:
            pass

        #checking if radius is greater than minimum radius
        if ( radius > 10):

            cv2.circle(image,(int(x),int(y)),int(radius),(255,0,0),2)
            cv2.circle(image,center,5,(0,0,255),-1)

    #pushing the points into the deque for plotting them afterwards
    pts.appendleft(center)
    for i in range(1,len(pts)):
        if((pts[i-1]==None) or (pts[i]==None) or (pts[5]==None) or( pts[-10]==None)):
            continue
        thickness=int(np.sqrt(64/float(i+1))*2.5)

        #drawing the redline
        cv2.line(image,pts[i-1],pts[i],(0,0,255),thickness)

        dX = pts[-10][0] - pts[i][0]
        dY = pts[-10][1] - pts[i][1]
        (dirX, dirY) = ("", "")
        direction=None

        #if the ball is moved significantly in the x direction
        if counter >= 10 and i == 1 and pts[-10] is not None:
            if np.abs(dX) > 20:
                dirX = "East" if np.sign(dX) == 1 else "West"

            #if the ball si moved significantly in the y direction
            if np.abs(dY) > 20:
                dirY = "North" if np.sign(dY) == 1 else "South"

            # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)

            # otherwise, only one direction is non-empty
            else:
                direction = dirX if dirX != "" else dirY
            cv2.putText(image, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 255), 3)
            cv2.putText(image, "dx: {}, dy: {}".format(dX, dY),
                        (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1)
    cv2.imshow("final image",image)
    counter+=1
    key = cv2.waitKey(30) & 0xff
    if(key==27):
        break

cv2.destroyAllWindows()