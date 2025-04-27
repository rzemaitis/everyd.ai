
import glob, sys
import numpy as np
import cv2
import utilities as util

def main(template,reviewdir,dimfile=None,width=None,height=None):
    template= template.replace('\\', '/')
    img_temp = cv2.imread(template)
    # Initialise dimensions
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

    # Rescale to given dimensions
    img_temp = util.rescale(img_temp,dim)
    detector, predictor = util.initDetectors()
    points_temp = util.findFace(img_temp, detector, predictor)
    # debug=False
    # if debug:
    #     for p in points_temp:
    #         x, y = p.astype(int)
    #         cv2.circle(img=img_temp, center=(x, y), radius=int(3*dim[1]/1200), color=(0, 255, 0), thickness=-1)
    #         cv2.imwrite(reviewdir + template.split('/')[-1], img_temp)
    #     sys.exit()

    np.savetxt(reviewdir+'template_'+str(dim[1])+'x'+str(dim[0])+'.txt', points_temp.astype(int))
    print('Template created successfully!')

if __name__ == "__main__":
    # Modifications to config formats
    reformat = {}
    reformat['int']=['width','height']
    reformat['addslash'] = ['reviewdir']
    # Read in config file
    config = util.readConfig('./config/templateMaker_config.txt', reformat=reformat)

    main(**config)
