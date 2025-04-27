import glob, sys, os
import numpy as np
from datetime import datetime,timedelta
import utilities as util

def main(imgdir,reviewdir,purgedir,datestart,datefinish,sleepstart=0,sleepfinish=0,extension='.jpg'):
    if sleepstart is None:
        sleepstart=0
    if sleepfinish is None:
        sleepfinish=0
    # Initialise dates and edit them according to sleep time
    datestart = datetime.strptime(datestart, '%Y-%m-%d').replace(hour=sleepstart)
    datefinish = datetime.strptime(datefinish, '%Y-%m-%d').replace(hour=sleepfinish)  + timedelta(days=1)
    if datefinish<datestart:
        print('FATAL ERROR: end date (datefinish) is earlier than the start date (datestart).')
        sys.exit()
    #Find all files within the given date range in the original directory
    dates_base,fnamelist_base=[],[]
    fnames = glob.glob(imgdir + '*'+ extension)
    if len(fnames)==0:
        sys.exit('No images found in'+imgdir+' with extension '+extension)
    for fname in fnames:
        fname = fname.replace('\\', '/')
        date=fname.split('/')[-1].rstrip(extension)
        date=datetime.strptime(date, "%Y-%m-%d_%H.%M.%S")
        if date<datefinish and date>datestart:
            dates_base.append(date)
            fnamelist_base.append(fname)

    # Find all dates within the given date range in the review directory
    dates_review, fnamelist_review=[],[]
    fnames = glob.glob(reviewdir + keyword)
    for fname in fnames:
        fname = fname.replace('\\', '/')
        date=fname.split('/')[-1].rstrip('_review'+extension)
        date=datetime.strptime(date, "%Y-%m-%d_%H.%M.%S")
        if date<datefinish and date>datestart:
            dates_review.append(date)
            fnamelist_review.append(fname)

    dates_base,dates_review=np.array(dates_base),np.array(dates_review)
    # dates_base = np.array(dates_base)
    days = (datefinish-datestart).days
    for i in range(days):
        # Construct dates running from the start date
        if i==0:
            testdate1 = datestart
            testdate2 = datestart.replace(hour=sleepfinish) + timedelta(days=1)
        else:
            testdate1 += timedelta(days=1)
            testdate2 += timedelta(days=1)
        #Check if there's only one photo left for this day in the review directory
        reviewcheck= dates_review[np.logical_and(dates_review>testdate1,dates_review<testdate2)]
        if reviewcheck.size > 1:
            print('Photos from',testdate1.strftime("%Y-%m-%d_%H.%M.%S"),'to',testdate2.strftime("%Y-%m-%d_%H.%M.%S"),
                  'still need reviewing.')
            continue
        #Find suitable days
        gooddates = dates_base[np.logical_and(dates_base>testdate1,dates_base<testdate2)]
        #Purge photos ONLY IF there's more than one on that day
        if gooddates.size != 1:
            for gooddate in gooddates:
                index = np.argwhere(dates_base == gooddate)[0][0]
                fname =fnamelist_base[index]
                sname= purgedir+fname.split('/')[-1]
                if gooddate not in dates_review:
                    print('Purging ' + fname.split('/')[-1])
                    os.rename(fname,sname)
                index_review = np.argwhere(dates_review == gooddate)[0][0]
                rname= fnamelist_review[index_review]
                os.remove(rname)
            # break
    print('Purging completed successfully!')

if __name__ == "__main__":
    # Modifications to config formats
    reformat = {}
    reformat['int']=['sleepstart','sleepfinish']
    reformat['addslash'] = ['imgdir','reviewdir','purgedir']
    # Read in config file
    config = util.readConfig('./config/purge_config.txt', reformat=reformat)

    main(**config)