import copy
import math
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
from typing import Iterable

import cv2
import dlib  # Facial recognition
import mediapipe as mp
import numpy as np


def init_face_detectors_legacy():
    # Load the detector
    detector = dlib.get_frontal_face_detector()

    # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor

def init_face_detectors():
    # Initialize MediaPipe once outside the function for efficiency
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    return face_mesh

def detect_face(img, face_mesh=None):
    """
    Detect face landmarks using MediaPipe Face Mesh.

    Args:
        img (np.array): BGR image (as read by cv2.imread)

    Returns:
        points (np.array): Array of shape (468, 2) with (x, y) pixel coordinates of landmarks.

    Raises:
        ValueError: if no face is detected.
    """

    if face_mesh is None:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    height, width, _ = img.shape
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        raise ValueError("No face detected")

    # If multiple faces, select the largest by bounding box area
    if len(results.multi_face_landmarks) > 1:
        max_area = 0
        selected_landmarks = None

        for face_landmarks in results.multi_face_landmarks:
            # Extract bounding box from landmarks
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Compute bounding box area in normalized coords
            area = (x_max - x_min) * (y_max - y_min)

            if area > max_area:
                max_area = area
                selected_landmarks = face_landmarks
    else:
        selected_landmarks = results.multi_face_landmarks[0]

    # Convert normalized landmarks to pixel coordinates
    points = np.zeros((468, 2))
    for i, lm in enumerate(selected_landmarks.landmark):
        x, y = lm.x * width, lm.y * height
        points[i] = [x, y]

    return points

def eye_points(points):
    """
    Extract eye landmarks from the full MediaPipe 468 landmarks.

    Args:
        points (np.array): Array of shape (468, 2) with (x, y) coordinates.

    Returns:
        points_eyes (np.array): Array of shape (N, 2) with all eye landmarks combined.
    """
    # MediaPipe eye landmark indices (left + right)
    eye_indices = [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,  # Left eye
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466  # Right eye
    ]
    
    points_eyes = points[eye_indices, :]
    return points_eyes


def detect_face_legacy(img, detector=None, predictor=None):
    # Taken from
    # https://towardsdatascience.com/detecting-face-features-with-python-30385aee4a8e
    # Initialise the detector and predictor
    if detector is None or predictor is None:
        detector, predictor = init_face_detectors()


    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)


    # Use detector to find landmarks
    faces = detector(gray)

    # TODO: If no face was found, raise error
    # if len(faces) == 0:
    #     return None

    # Initialise with assumption that there is only one face
    goodfaceid = 0
    # Failsafe if multiple faces were found - choose the largest face
    if len(faces) > 1:
        maxdist = 0.0
        for i, face in enumerate(faces):
            # Check face size
            dist = distance(face.left(), face.top(),
                            face.right(), face.bottom())
            if dist > maxdist:
                maxdist = dist
                goodfaceid = i

    # Save points from the selected face
    landmarks = predictor(image=gray, box=faces[goodfaceid])
    points = np.ones((68, 2))
    for n in range(0, 68):
        points[n][0], points[n][1] = landmarks.part(n).x, landmarks.part(n).y
    return points


def rescale(img, dim):
    # Border fill taken from
    # https://stackoverflow.com/questions/11142851/adding-borders-to-an-image-using-python
    # (Ans 3)
    # Cropping taken from:
    # https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
    h_im, w_im = img.shape[:2]
    h_temp, w_temp = dim
    # If given image is larger than either of the dimensions
    if h_im > h_temp or w_im > w_temp:
        newscale = np.hypot(h_temp, w_temp) / np.hypot(h_im, w_im)
        img = cv2_clipped_zoom(img, newscale)
        # # Find which dimension in lower in ratio
        # w_ratio, h_ratio = w_im / w_temp, h_im / h_temp
        # # Scale the image down
        # kwargs = {}
        # if w_ratio > h_ratio:
        #     kwargs['height'] = h_temp
        # else:
        #     kwargs['width'] = w_temp
        # img = rescale_pristine(img, **kwargs)
        # Renew dimensions
        h_im, w_im = img.shape[:2]
    # Black borders
    color = [0, 0, 0]
    # Width first
    if h_im > h_temp:
        img = img[
            int((h_im - h_temp) / 2): -(h_im - h_temp) +
            + int((h_im - h_temp) / 2), :
        ]
    elif h_im < h_temp:
        img = cv2.copyMakeBorder(
            img,
            int((h_temp - h_im) / 2),
            (h_temp - h_im) - int((h_temp - h_im) / 2),
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=color,
        )
    # Height
    if w_im > w_temp:
        img = img[
            :, int((w_im - w_temp) / 2): -(w_im - w_temp) +
            + int((w_im - w_temp) / 2)
        ]
    elif w_im < w_temp:
        img = cv2.copyMakeBorder(
            img,
            0,
            0,
            int((w_temp - w_im) / 2),
            (w_temp - w_im) - int((w_temp - w_im) / 2),
            cv2.BORDER_CONSTANT,
            value=color,
        )

    return img


def rescale_pristine(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Resize the image while keeping the aspect ratio
    # Taken from
    # https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    # Grab the image size
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
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def distance(x, y, x1, y1):
    return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)


def landmark_transform(points, dim, params):
    """
    Transform the landmark points according to given parameters.
    The parameters are - x and y offsets, scale and rotation.

    Other possible parameters could include (I will not implement these):
    xscale, yscale - better fit, but distorts the original image.
    2D plate fit - solves the imperfect face pointing problem,
    but introduces obvious distortion
    and probably needs some special AI algorithm to fit.
    """

    # Parameters used below: x offset, y offset, scale, rotation
    xoff, yoff, scale, rotation = params
    # Dim is ylength, xlength
    scaled = copy.copy(points)
    # Find midpoint of image
    midpoint = (dim[1] * 0.5, dim[0] * 0.5)
    rotation = np.radians(rotation)
    scaled[:, 0] += xoff
    scaled[:, 1] += yoff
    # Scale the data
    # Both the sign of the difference and the scale takes care of everything
    xscales = (scaled[:, 0] - midpoint[0]) * (scale - 1)
    yscales = (scaled[:, 1] - midpoint[1]) * (scale - 1)
    scaled[:, 0] += xscales
    scaled[:, 1] += yscales
    # Make midpoint the origin
    scaled[:, 0] -= midpoint[0]
    scaled[:, 1] -= midpoint[1]
    # Get length of all vectors
    # From
    # https://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
    dist = np.sqrt((scaled**2).sum(-1))
    # Find angle of each facepoint
    angles = calc_angle(scaled)
    # Rotate each point by given rotation
    scaled[:, 0] = dist * np.cos(-angles - rotation)
    scaled[:, 1] = dist * np.sin(-angles - rotation)
    # Recenter according to midpoint
    scaled[:, 0] += midpoint[0]
    scaled[:, 1] += midpoint[1]

    return scaled


def calc_angle(vectors):
    # Make zero angle vector
    zvector = (1, 0)
    # From
    # https://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
    mag_v = np.sqrt((vectors**2).sum(-1))
    angle = np.arccos(np.dot(vectors, zvector) / mag_v)
    vectangle = -1 * np.arcsin(np.cross(vectors, zvector) / mag_v)

    angle = np.where(vectangle > 0.0, -1 * angle, angle)
    if len(angle) == 1:
        angle = angle[0]
    return angle


def costfunction(params, points, template_points, dim):
    # Convert points to scales
    scaled_f = landmark_transform(points, dim, params)
    # Calculate distance from points
    cost = np.sum(
        distance(scaled_f[:, 0], scaled_f[:, 1],
                 template_points[:, 0], template_points[:, 1])
    )
    return cost


def costfunction_nooffset(points, template_points):
    # Calculate distance from points
    cost = np.sum(
        distance(points[:, 0], points[:, 1],
                 template_points[:, 0], template_points[:, 1])
    )
    return cost


def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image
    and returning an enlarged/shrinked view of
    the image without changing dimensions
    Taken from
    https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
    (Ans 2)
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    Note:
        Bigger number - bigger image
    """
    h, w = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(h * zoom_factor), int(w * zoom_factor)

    # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size
    # in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - h) // 2, max(0, new_width - w) // 2
    y2, x2 = y1 + h, x1 + w
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, h), min(new_width, w)
    pad_height1, pad_width1 = (h - resize_height) // 2, (w - resize_width) // 2
    pad_height2, pad_width2 = (h - resize_height) - pad_height1, (
        w - resize_width
    ) - pad_width1
    pad_spec = [(pad_height1, pad_height2),
                (pad_width1, pad_width2)] + [(0, 0)] * (
        img.ndim - 2
    )

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode="constant")
    assert result.shape[0] == h and result.shape[1] == w
    return result


def image_add_date(img, text, fontcolor=(0, 255, 0)):
    # Taken from and improved:
    # https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    ylength, xlength = img.shape[:2]  # y,x
    position = (int(xlength * 0.75), int(ylength * 0.97))
    # position = (int(xlength * 0.65), int(ylength * 0.97))
    # fontscale and linetype have been meticulously tested for these values
    fontscale = xlength / 10e2
    linetype = math.ceil((xlength / 10e2) * 2.5)
    cv2.putText(img, text, position, font, fontscale, fontcolor, linetype)
    return img


def fname_to_date(fnames: Iterable[Path], dateformat="%Y-%m-%d_%H.%M.%S") -> np.ndarray:
    dates = []
    for fname in fnames:
        # fname is already a Path object
        stem = fname.stem
        # Remove any suffix like _review, _test, etc.
        cleaned_stem = re.sub(r'_[a-zA-Z]+$', '', stem)
        try:
            date = datetime.strptime(cleaned_stem, dateformat)
            dates.append(date)
        except ValueError as e:
            raise ValueError(
                f"Failed to parse date from filename '{fname}' using format '{dateformat}'."
            ) from e
    return np.array(dates)


def rotate_image(img, angle):
    # Taken from https://www.pyimagesearch.com/2021/01/20/opencv-rotate-image/
    # grab the dimensions of the image and calculate the center of the
    # image
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # rotate our image by 45 degrees around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))
    return img


def translate_image(img, xoff: float, yoff: float):
    # Taken from
    # https://stackoverflow.com/questions/54274185/
    # shifting-an-image-by-x-pixels-to-left-while-maintaining-the-original-shape
    # Grab the dimensions of the image
    (h, w) = img.shape[:2]

    # Image translation
    translation_matrix = np.array([[1, 0, xoff], [0, 1, yoff]], dtype=np.float32)
    img = cv2.warpAffine(img, translation_matrix, (w, h))
    return img


def create_dir(dirname: str, parent_dir: str = "./") -> None:
    """
    Creates a directory at parent_dir/dirname if it does not already exist.

    Parameters:
    -----------
    dirname : str
        Name of the directory to create.
    parent_dir : str, optional
        Path to the parent directory (default is current directory).

    Returns:
    --------
    None
    """
    path = Path(parent_dir) / dirname
    path.mkdir(parents=True, exist_ok=True)


def dir_slash(dirs):
    for k in dirs.keys():
        dirs[k] = os.path.join(os.path.normpath(dirs[k]), "").replace(os.sep, "/")
    return dirs


def read_image(fname):
    """Read image and convert to RGB"""
    img = cv2.imread(str(fname))
    if img is None:
        raise FileNotFoundError(f"Image {fname} not found.")
    return img


def write_image(fname, img):
    """Write image to file"""
    cv2.imwrite(str(fname), img)


def image_finder(config_main, directory, sleepstart=0):
    """
    Finds all images in the specified directory and filters them by date.
    Returns both filenames and dates of the images.
    """
    # Find all images in the directory
    # TODO Path has its own glob generator, use that
    directory_path = Path(directory)
    fnames = list(directory_path.glob(f"*{config_main['extension']}"))

    # Cull by date
    datemask = np.ones(len(fnames), dtype=bool)
    dates = fname_to_date(fnames)
    if config_main['datestart'] != '':
        try:
            datestart = datetime.strptime(config_main['datestart'], '%Y-%m-%d') + timedelta(hours=sleepstart)
        except ValueError:
            print("TODO: better errors. This one failed when converting a date.")
            raise
        datemask = datemask & (dates > datestart)
    if config_main['datefinish'] != '':
        try:
            datefinish = datetime.strptime(config_main['datefinish'], '%Y-%m-%d'
                                           ) + timedelta(days=1) + timedelta(hours=sleepstart)
        except ValueError:
            print("TODO: better errors. This one failed when converting a date.")
            raise
        datemask = datemask & (dates < datefinish)

    # Choose picture names by date if we need to
    if not all(datemask):
        fnames = [fnames[i] for i in np.where(datemask)[0]]
        dates = [dates[i] for i in np.where(datemask)[0]]

    # In the case of no images found
    if len(fnames) == 0:
        raise FileNotFoundError(
            f"No images found in {directory} "
            f"with extension {config_main['extension']}. "
            "Try changing your selected date range."
        )
    # TODO: np.array is redundant once the above todos are fixed
    return fnames, np.array(dates)


def daily_dates(fnames, sleepstart=0):
    """
    Turn image file names into strings
    of dates compliant with the solution
    files format.
    """
    dates = fname_to_date(fnames)
    for i, date in enumerate(dates):
        # If time the photo was taken is before
        # designated sleep time, subtract a day.
        if date.hour < sleepstart:
            date -= timedelta(days=1)
        dates[i] = date.strftime('%Y-%m-%d')
    return np.array(dates)
