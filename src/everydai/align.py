import sys
import cv2
from scipy.optimize import minimize
import utilities as util
import numpy as np
import glob
from scipy.ndimage import rotate
from datetime import datetime,timedelta
import multiprocessing as mp
from functools import partial

def solver(img,detectors,dim_old,points_temp,oriented=True):
    dim = img.shape[:2]
    detector, predictor = detectors
    # Detection
    points = util.findFace(img, detector, predictor)
    # Failsafe if no faces were found
    if points is None:
        return None

    # Null-hypothesis guess
    guess = np.array([0.0, 0.0, np.hypot(dim[0],dim[1]) / np.hypot(dim_old[0],dim_old[1]), 0.])
    bounds = ((-dim[1], dim[1]), (-dim[0], dim[0]), (guess[2] / 5., guess[2] * 5.), (-90., 90.))
    if not oriented:
        bounds[3] = (-180., 180.)
    res = minimize(util.costFunction, guess, args=(points, points_temp, dim), bounds=bounds)
    return res.x

def saveImage(img,sname,params):
    xoff, yoff, scale, angle = params

    # The number of pixels
    dim = img.shape[:2]

    # Image translation
    # Taken from https://stackoverflow.com/questions/54274185/shifting-an-image-by-x-pixels-to-left-while-maintaining-the-original-shape
    translation_matrix = np.float32([[1, 0, xoff], [0, 1, yoff]])
    img = cv2.warpAffine(img, translation_matrix, (dim[1], dim[0]))

    # Scale image
    img = util.cv2_clipped_zoom(img, scale)

    # Rotate image
    # Taken from https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
    img = rotate(img, angle, reshape=False)

    print('Image done:', sname.split('/')[-1].replace('_movie',''))
    cv2.imwrite(sname, img)

def align(img_temp,detectors,points_temp,moviedir,errorlog,fname,oriented=True):
    fname_crop = fname.replace('\\', '/').split('/')[-1]  # Windows changes the last / to \
    fname_movie = moviedir + fname_crop.rpartition('.')[0] + '_movie.' + fname_crop.rpartition('.')[-1]
    img = cv2.imread(fname)
    dim_old = img.shape[:2]  # Used for initial guess for the solution
    img = util.rescale(img, img_temp.shape[:2])

    params = solver(img, detectors, dim_old, points_temp, oriented=oriented)
    if params is None:
        print('No face found in file ' + fname_crop)
        errorlog.write('No face found in file ' + fname_crop)
        return
    saveImage(img, fname_movie, params)


def main(template,imgdir,moviedir,extension='.jpg',datestart=None,datefinish=None,oriented=True,cores=1,dimfile=None,width=None,height=None):
    '''
    :param keyword: Name of your selected images with wildcards (*)
    :param fname_temp: Name of your template (make sure it's your largest image)
    :param imgdir: Image directory
    :param savedir: Saving directory
    (WARNING: make sure it's not the same as image directory, as files will be overwritten)
    :return:
    '''
    #Read in and reshape template image if needed
    img_temp = cv2.imread(template)
    #Initialise dimensions
    if dimfile is not None:
        try:
            dim = cv2.imread(dimfile).shape[:2]
        except:
            print('Dimension image not found in',dimfile)
            sys.exit()
    # Supplied dimensions
    elif width is not None and height is not None:
        dim=(height,width)
    #No dimesnions supplied: use the template image
    else:
        try:
            dim = img_temp.shape[:2]
        except:
            print('Template image not found in',img_temp)
            sys.exit()
    if dim!=img_temp.shape[:2]:
        img_temp=util.rescale(img_temp,dim)
    fnames = glob.glob(imgdir+'*'+extension)
    #Choose filenames by date
    dates = util.toDate(fnames)
    if datestart is not None and datefinish is not None:
        datestart, datefinish = datetime.strptime(datestart, '%Y-%m-%d'), datetime.strptime(datefinish, '%Y-%m-%d') + timedelta(days=1)
    fnames = [fnames[i] for i in np.where(np.logical_and(dates>datestart,dates<datefinish))[0]]
    if len(fnames)==0:
        print('No images found in'+imgdir+' with extension '+extension)
        if datestart is not None and datefinish is not None:
            print('Try changing your selected date range.')
        sys.exit()

    #Initialise other things for aligning
    detectors = util.initDetectors()
    points_temp = util.findFace(img_temp, detectors[0], detectors[1])
    kwargs = {}
    kwargs['oriented']=oriented
    errorlog = open(moviedir + 'badphotos_align.txt', "w")

    alignpartial = partial(align,img_temp,detectors,points_temp,moviedir,errorlog,**kwargs)
    if cores==1:
        for fname in fnames:
            alignpartial(fname)
    else:
        pool = mp.Pool(processes=cores)
        pool.map(alignpartial,fnames)
        pool.close()
        pool.join()
    errorlog.close()
    print('Aligning finished successfully!')

if __name__ == "__main__":
    #Modifications to config formats
    reformat={}
    reformat['int']=['cores','width','height']
    reformat['bool'] = ['oriented']
    reformat['addslash']=['imgdir','moviedir']
    #Read in config file
    config =  util.readConfig('./config/align_config.txt',reformat=reformat)

    main(**config)