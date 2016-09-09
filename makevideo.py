import cv2
import os

vvw           =   cv2.VideoWriter('GentImmune.avi',cv2.VideoWriter_fourcc('X','V','I','D'),
                                  24,(740,880))
#VVW = cv2.VideoWriter('home/sami/Desktop/code/segmentation/immune_partner/contour/video.avi',-1,1,(740,880))

frameslist    =   os.listdir('/home/sami/Desktop/code/segmentation/immune_partner/contour/')
howmanyframes =   len(frameslist)
print('Frames count: '+str(howmanyframes)) #just for debugging


for i in range(0,howmanyframes):
    print(i)
    theframe = cv2.imread('/home/sami/Desktop/code/segmentation/immune_partner/contour/'+frameslist[i])
    VVW.write(theframe)
cv2.destroyAllWindows()
VVW.release()