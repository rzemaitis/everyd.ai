import glob, sys
import numpy as np
import cv2
import utilities as util
from datetime import datetime,timedelta

def main(imgdir,reviewdir,datestart,datefinish,sleepstart=0,sleepfinish=0,puttemplate=False,extension='.jpg'):
    #Boolean to check if we need to store photo names taken at unusual time
    needlog=False
    if sleepstart is None:
        sleepstart=0
    if sleepfinish is None:
        sleepfinish=0
    # Initialise dates and edit them according to sleep time
    datestart = datetime.strptime(datestart, '%Y-%m-%d').replace(hour=sleepstart)
    datefinish = datetime.strptime(datefinish, '%Y-%m-%d').replace(hour=sleepfinish) + timedelta(days=1)

    if datefinish<datestart:
        print('FATAL ERROR: end date (datefinish) is earlier than the start date (datestart).')
        sys.exit()
    #Find all files within the given date range
    dates,fnamelist=[],[]
    fnames = glob.glob(imgdir + '*'+ extension)
    if len(fnames)==0:
        sys.exit('No images found in'+imgdir+' with extension '+extension)
    for fname in fnames:
        fname = fname.replace('\\','/')
        date=fname.split('/')[-1].rstrip(extension)
        date=datetime.strptime(date, "%Y-%m-%d_%H.%M.%S")
        if date<datefinish and date>datestart:
            dates.append(date)
            fnamelist.append(fname)
    dates=np.array(dates)

    #Initialise parameters
    dim = (0,0)
    days = (datefinish-datestart).days
    # Construct dates running from the start date
    testdate1 = datestart
    testdate2 = datestart.replace(hour=sleepfinish) + timedelta(days=1)
    needsreview = 0  # Counter to check how many dates in the given ranges have not been reviewed yet
    for i in range(days):
        # Find suitable days
        gooddates = dates[np.logical_and(dates > testdate1, dates < testdate2)]
        #Put photos up for review ONLY IF there's more than one on that day
        if gooddates.size >1:
            #Change flag that checks if all dates have only one picture
            needsreview+=1
            for gooddate in gooddates:
                index = np.argwhere(dates == gooddate)[0][0]
                fname = fnamelist[index]
                img = cv2.imread(fname)
                sname = reviewdir + fname.rpartition('.')[0].split('/')[-1] + '_review.' + fname.rpartition('.')[-1]
                # Put this picture to log to check why it's taken at a weird time
                if gooddate.hour > sleepstart and gooddate.hour < sleepfinish:
                    print('Unusual photo time found in', testdate1.strftime('%Y-%m-%d'))
                    if not needlog:
                        needlog=True
                        log = open(reviewdir+'unusualtime.txt', "w")
                    log.write(fname.split('/')[-1]+'\n')


                #Put a template on top of image and copy it
                if puttemplate:
                    # If dimensions are the same as the last picture
                    if dim == img.shape[:2]:
                        pass
                    else:
                        dim = img.shape[:2]
                        try:
                            points =np.loadtxt(reviewdir+'template_'+str(dim[1])+'x'+str(dim[0])+'.txt')
                        except:
                            print('File template_'+str(dim[1])+'x'+str(dim[0])+'.txt not found in '+reviewdir)
                            print('Please set puttemplate=False or run this first:')
                            print('templateMaker.py <templateimage(full path)> <reviewdirectory> width='+str(dim[1])+' height='+str(dim[0]))
                            sys.exit()
                    for p in points:
                        x, y = p.astype(int)
                        circlesize = int(3 * np.hypot(dim[0], dim[1]) / 1200) #Scaled by the diagonal
                        cv2.circle(img=img, center=(x, y), radius=circlesize, color=(0, 255, 0),
                                   thickness=-1)
                img = util.imageText(img, testdate1.strftime('%Y-%m-%d'))
                cv2.imwrite(sname, img)
        testdate1 += timedelta(days=1)
        testdate2 += timedelta(days=1)
    if needsreview==0:
        print('All dates have already been reviewed from',datestart.strftime('%Y-%m-%d'),'to',(datefinish-timedelta(days=1)).strftime('%Y-%m-%d')+'.')
    elif needsreview==1:
        print(str(needsreview),'day still needs reviewing from',datestart.strftime('%Y-%m-%d'),'to',(datefinish-timedelta(days=1)).strftime('%Y-%m-%d')+'.')
    else:
        print(str(needsreview), 'days still need reviewing from', datestart.strftime('%Y-%m-%d'), 'to', (datefinish-timedelta(days=1)).strftime('%Y-%m-%d')+'.')
    if needlog:
        log.close()
    print('Review photos created successfully!')

if __name__ == "__main__":
    # Modifications to config formats
    reformat = {}
    reformat['int']=['sleepstart','sleepfinish']
    reformat['bool'] = ['puttemplate']
    reformat['addslash'] = ['imgdir','reviewdir']
    # Read in config file
    config = util.readConfig('./config/review_config.txt', reformat=reformat)

    main(**config)