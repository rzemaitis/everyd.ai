import dlib
import numpy as np
import cv2,copy
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
    if len(faces)>1:
        print('Problem')
    # Save points
    points=np.ones((68,2))
    for face in faces:
        # Look for the landmarks
        landmarks = predictor(image=gray, box=face)
        # Loop through all the points
        for n in range(0, 68):
            points[n][0],points[n][1] = landmarks.part(n).x, landmarks.part(n).y
    return points

def rescale(img, dim, inter = cv2.INTER_AREA):
    #Border fill taken from https://stackoverflow.com/questions/11142851/adding-borders-to-an-image-using-python (Ans 3)
    #Cropping taken from: https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
    h_im,w_im= img.shape[:2]
    h_temp, w_temp = dim
    #Black filling
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
# def rescale(image, dim, inter = cv2.INTER_AREA):
    ##Taken from https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    # initialize the dimensions of the image to be resized and
    # grab the image size
    # height,width = dim
    # dim = None
    # (h, w) = image.shape[:2]
    #
    # # if both the width and height are None, then return the
    # # original image
    # if width is None and height is None:
    #     return image
    #
    # # check to see if the width is None
    # if width is None:
    #     # calculate the ratio of the height and construct the
    #     # dimensions
    #     r = height / float(h)
    #     dim = (int(w * r), height)
    #
    # # otherwise, the height is None
    # else:
    #     # calculate the ratio of the width and construct the
    #     # dimensions
    #     r = width / float(w)
    #     dim = (width, int(h * r))

    # # resize the image
    # resized = cv2.resize(image, dim, interpolation = inter)
    #
    # # return the resized image
    # return resized
# def rescale(fname, dim):
#     '''
#     Rescale images using PIL
#     Taken from https://stackoverflow.com/questions/11142851/adding-borders-to-an-image-using-python
#     :param fname:
#     :param dim:
#     :return:
#     '''
#     old_im = Image.open(fname)
#     old_size = old_im.size
#
#     new_size = (800, 800)
#     new_im = Image.new("RGB", new_size)  ## luckily, this is already black!
#     new_im.paste(old_im, ((new_size[0] - old_size[0]) // 2,
#                           (new_size[1] - old_size[1]) // 2))
#
#     # new_im.show()
#     # new_im.save('someimage.jpg')
#     return new_im

def distance(x,y,x1,y1):
    return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)

def offsetScale(points,dim,params):
    #Dim is xlength, ylength
    scaled=copy.copy(points)
    #Find midpoint of dimensions
    midpoint=(dim[0] * 0.5, dim[1] * 0.5)
    #x offset, y offset, scale
    xoff, yoff, scale, rotation = params
    rotation = np.radians(rotation)
    scaled[:,0] +=xoff
    scaled[:, 1] += yoff
    #Scale the data
    if scale<1:
        scale = (scale-1)
    else:
        scale = scale-1
    #Both the sign of the difference and the scale takes care of everything
    xscales = (scaled[:,0]-midpoint[0])*scale
    yscales = (scaled[:,1]-midpoint[1])*scale
    scaled[:,0] += xscales
    scaled[:,1] += yscales
    #TODO: maybe also include rotation?
    #dist = distance(scaled[:,0],scaled[:,1],midpoint[0],midpoint[1])
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

def midpoint(dim):
    return dim[0] * 0.5, dim[1] * 0.5