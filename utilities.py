import dlib #Facial recognition
import math
import numpy as np
import cv2,copy
from datetime import datetime
def initDetectors():
    # Load the detector
    detector = dlib.get_frontal_face_detector()

    # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor

def findFace(img,detector,predictor):
    #Taken from https://towardsdatascience.com/detecting-face-features-with-python-30385aee4a8e
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

    #If no face was found, return None
    if len(faces)==0:
        return None
    #Initialise with assumption that there is only one face
    goodfaceid=0
    # Failsafe if multiple faces were found - choose the biggest face
    if len(faces)>1:
        maxdist=0.
        for i,face in enumerate(faces):
            #Check face side distance 1-17
            dist = distance(face.left(),face.top(),face.right(),face.bottom())
            if dist>maxdist:
                maxdist=dist
                goodfaceid = i
    # Save points from the biggest face
    landmarks = predictor(image=gray, box=faces[goodfaceid])
    points = np.ones((68, 2))
    for n in range(0, 68):
        points[n][0], points[n][1] = landmarks.part(n).x, landmarks.part(n).y
    return points

def rescale(img, dim):
    #Border fill taken from https://stackoverflow.com/questions/11142851/adding-borders-to-an-image-using-python (Ans 3)
    #Cropping taken from: https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
    h_im,w_im= img.shape[:2]
    h_temp, w_temp = dim
    #If given image is larger than the dimensions
    if h_im>h_temp or w_im>w_temp:
        #Find which dimension in higher in ratio
        w_ratio,h_ratio = w_im/w_temp,h_im/h_temp
        # Scale the image down
        kwargs = {}
        if w_ratio<h_ratio:
            kwargs['height'] = h_temp
        else:
            kwargs['width'] = w_temp
        img = rescalePristine(img,**kwargs)
        #Renew dimensions
        h_im, w_im = img.shape[:2]
    #Black borders
    color=[0,0,0]
    #Width first
    if h_im>h_temp:
        img=img[int((h_im-h_temp)/2):-(h_im-h_temp)+int((h_im-h_temp)/2),:]
    elif h_im<h_temp:
        img = cv2.copyMakeBorder(img, int((h_temp-h_im)/2), (h_temp-h_im)-int((h_temp-h_im)/2), 0, 0,  cv2.BORDER_CONSTANT, value=color)
    #Height
    if w_im>w_temp:
        img=img[:,int((w_im-w_temp)/2):-(w_im-w_temp)+int((w_im-w_temp)/2)]
    elif w_im<w_temp:
        img = cv2.copyMakeBorder(img, 0, 0, int((w_temp-w_im)/2), (w_temp-w_im)-int((w_temp-w_im)/2), cv2.BORDER_CONSTANT, value=color)

    return img

def rescalePristine(image, width = None, height = None, inter = cv2.INTER_AREA):
    #Resize the image while keeping the aspect ratio
    # Taken from https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

def distance(x,y,x1,y1):
    return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)

def offsetScale(points,dim,params):
    # Other possible parameters could include (I will not implement these):
    # xscale, yscale - better fit, but distorts the original image.
    # 2D plate fit - solves the imperfect face pointing problem,
    # but introduces obvious distortion and probably needs some special AI algorithm to fit.
    # Parameters used below: x offset, y offset, scale, rotation
    xoff, yoff, scale, rotation = params
    # Dim is ylength, xlength
    scaled=copy.copy(points)
    # Find midpoint of image
    midpoint=(dim[1] * 0.5, dim[0] * 0.5)
    rotation = np.radians(rotation)
    scaled[:,0] +=xoff
    scaled[:, 1] += yoff
    # Scale the data
    #Both the sign of the difference and the scale takes care of everything
    xscales = (scaled[:,0]-midpoint[0])*(scale-1)
    yscales = (scaled[:,1]-midpoint[1])*(scale-1)
    scaled[:,0] += xscales
    scaled[:,1] += yscales
    #Make midpoint the origin
    scaled[:,0]-=midpoint[0]
    scaled[:,1]-=midpoint[1]
    #Get length of all vectors
    # From https://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
    dist = np.sqrt((scaled ** 2).sum(-1))
    #Find angle of each facepoint
    angles = calcAngle(scaled)
    #Rotate each point by given rotation
    scaled[:,0]=dist*np.cos(-angles-rotation)
    scaled[:,1]=dist*np.sin(-angles-rotation)
    #Recenter according to midpoint
    scaled[:,0]+=midpoint[0]
    scaled[:, 1] += midpoint[1]

    return scaled

def calcAngle(vectors):
    # Make zero angle vector
    zvector = (1, 0)
    # From https://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
    mag_v= np.sqrt((vectors ** 2).sum(-1))
    angle = np.arccos(np.dot(vectors, zvector) / mag_v )
    vectangle = -1 * np.arcsin(np.cross(vectors, zvector) / mag_v )

    angle = np.where(vectangle > 0., -1 * angle, angle)
    if len(angle) == 1:
        angle = angle[0]
    return angle

def costFunction(params,points_f,points_temp,dim):
    #Convert points to scales
    scaled_f = offsetScale(points_f,dim,params)
    #Calculate distance from points
    cost = np.sum(distance(scaled_f[:,0],scaled_f[:,1],points_temp[:,0], points_temp[:,1]))
    return cost

def costFunction_nooffset(points_f,points_temp,dim):
    #Calculate distance from points
    cost = np.sum(distance(points_f[:,0],points_f[:,1],points_temp[:,0], points_temp[:,1]))
    return cost

def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Taken from https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions (Ans 2)
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def imageText(img,text):
    #Taken from and improved: https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    ylength,xlength = img.shape[:2] #y,x
    position = (int(xlength*0.75), int(ylength*0.97))
    #fontScale and lineType have been meticulously tested for these values
    fontScale = (xlength/10e2)
    fontColor = (0, 255, 0)
    lineType = math.ceil((xlength/10e2)*2.5)
    cv2.putText(img, text,
                position,
                font,
                fontScale,
                fontColor,
                lineType)
    return img

def toDate(fnames):
    dates=[]
    for fname in fnames:
        fname = fname.replace('\\', '/').split('/')[-1]
        date = fname.split('/')[-1].rstrip('.jpg')
        dates.append(datetime.strptime(date, "%Y-%m-%d_%H.%M.%S"))
    return np.array(dates)

def readConfig(fname,reformat=None):
    with open(fname) as configfile:
        config={}
        lines = filter(None, (line.rstrip().replace(" ", "") for line in configfile))
        for line in lines:
            if not line.startswith('#'):
                argname,argvalue=line.split('=')
                config[argname]=argvalue
                if config[argname] == 'None':
                    config[argname] = None
                #Predefined modifications
                if reformat is not None:
                    #Change to int
                    if 'int' in list(reformat.keys()):
                        if argname in reformat['int'] and config[argname] is not None:
                            config[argname] = int(config[argname])
                    #Change to bool
                    if 'bool' in list(reformat.keys()):
                        if argname in reformat['bool'] and config[argname] is not None:
                            config[argname] = bool(config[argname])
                    #Add slash to directories if missing
                    if 'addslash' in list(reformat.keys()):
                        if argname in reformat['addslash'] and config[argname] is not None:
                            config[argname] = config[argname].replace('\\', '/')
                            if config[argname][-1] != '/':
                                config[argname] += '/'
    return config