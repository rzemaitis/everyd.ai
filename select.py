import sys
import cv2
from scipy.optimize import minimize
import utilities as util
import numpy as np
import glob
from scipy.ndimage import rotate
def main(keyword,fname_temp,imgdir,savedir):
    '''
    :param keyword: Name of your selected images with wildcards (*)
    :param fname_temp: Name of your template (make sure it's your largest image)
    :param imgdir: Image directory
    :param savedir: Saving directory
    (WARNING: make sure it's not the same as image directory, as files will be overwritten)
    :return:
    '''
    img_temp = cv2.imread(imgdir + fname_temp)
    fnames = glob.glob(imgdir+keyword)
    detector, predictor = util.initDetectors()
    print(imgdir + fname_temp)
    points_temp = util.findFace(img_temp, detector, predictor)
    for fname in fnames:
        fname_crop = fname.split('\\')[-1]
        img = cv2.imread(fname)
        img = util.rescale(img, img_temp.shape[:2])
        dim = np.flip(img_temp.shape[:2]) # I work with [width,height]
        #file1_resc.show()
        ########################
        #Detection
        ######################

        #Save points
        points = util.findFace(img,detector,predictor)
        #Failsafe if no faces were found
        if np.unique(points).size==1:
            print('No face found in file '+fname_crop)
            continue
        debug=False
        if debug:
            for i, p in enumerate(points):
                # x_orig,y_orig=points_1[i].astype(int)
                x, y = p.astype(int)
                # print(x_orig,y_orig,x,y)
                # print(x,y)
                cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
            cv2.imwrite(savedir + fname.split('\\')[-1], img)
            sys.exit()
        #Null-hypothesis guess
        guess = np.array([0.0,0.0,1.0,0.])
        bounds=((None,None),(None,None),(0.01,None),(-180.,180))
        res=minimize(util.costFunction,guess,args=(points,points_temp,dim),bounds=bounds)
        xoff,yoff,scale,angle = res.x
        # xoff, yoff, scale, angle = guess
        scaled = util.offsetScale(points,dim,[xoff,yoff,scale,angle])

        # The number of pixels
        num_rows, num_cols = img.shape[:2]

        # Creating a translation matrix (the y offset has to be flipped)
        #Taken from https://stackoverflow.com/questions/54274185/shifting-an-image-by-x-pixels-to-left-while-maintaining-the-original-shape
        translation_matrix = np.float32([[1, 0, xoff], [0, 1, yoff]])

        # Image translation
        img = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

        #Scale image
        img=util.cv2_clipped_zoom(img,scale)

        #Rotate image
        # Taken from https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
        img = rotate(img, angle, reshape=False)


        # cv2.imwrite(savedir + 'test.jpg', img_1)
        #OVerlay for check
        # Draw points
        debug=False
        if debug:
            for i,p in enumerate(scaled):
                # x_orig,y_orig=points_1[i].astype(int)
                x,y = p.astype(int)
                # print(x_orig,y_orig,x,y)
                print(x,y)
                cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
        #added_image = cv2.addWeighted(img_translation, 1.0, img_temp, 0.1, 0)
        # show the image
        #cv2.imshow(winname="Face", mat=img_1)
        print('Saving:', fname_crop)
        cv2.imwrite(savedir + fname_crop, img)
        #print(dim,(dim[0] * 0.5, dim[1] * 0.5))
        # Wait for a key press to exit
        #cv2.waitKey(delay=0)


if __name__ == "__main__":
    keyword = sys.argv[1]
    fname_temp = sys.argv[2]
    imgdir = sys.argv[3]
    savedir = sys.argv[4]
    main(keyword,fname_temp,imgdir,savedir)