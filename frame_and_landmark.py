# import the required library
import numpy as np
import dlib
import cv2
 


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
 
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

 
    # return the list of (x, y)-coordinates
    return coords



from collections import OrderedDict
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])





     
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
     
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
     

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('psesst.mp4')



#f = open('bbb.txt', 'w+') #w = write, + is for creating the file if it doesn't exist
result_array = np.empty((0,2), int)

frame_number = 0




while(cap.isOpened()):
    # Capture frame-by-frame
       
    ret, image = cap.read()

    if ret == False:
        print('cannot access the frame')
        break

    else:
        frame_number += 1
        print('frame number is ',frame_number)

    
     
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         
    rects = detector(gray, 1)

    #result = [] #native list
    #result_array = np.array([])
    #result_array = np.empty((0,2), int)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
             
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

    
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
     
        (x, y, w, h) = rect_to_bb(rect)
     
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image


        if len(shape) == 0: #=empty
           print('shape is empty') #instructions if shape is empty here
        else:
            print(len(shape))



        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1) 
            #result.append([x, y]) #adding each element of shape in result array
            result_array = np.vstack((result_array, (x, y)))
    
    #f.write(str(result))
 

    #Display the resulting frame
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('video is finished')
        break






#np.save('psesst.dat', result_array)

#np.save('katemu.dat', result_array)

#f.write(str(result)) #writing the whole result in the file (write method accepts only strings, that's why I used str())
     
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()