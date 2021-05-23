import glob, os, sys
from PIL import Image, ExifTags
from datetime import datetime
import utilities as util

def main(imgdir,extension='.jpg'):
    errorlog = open(imgdir + 'errors_renamePhotos.txt', "w")
    fnames = glob.glob(imgdir + '*'+extension)
    if len(fnames) == 0:
        sys.exit('No images found in ' + imgdir + ' with extension ' + extension)
    for fname in fnames:

        with Image.open(fname) as img:
            # Taken from https://stackoverflow.com/questions/21697645/how-to-extract-metadata-from-a-image-using-python
            try:
                exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}
            except:
                print('Problem with '+fname.split('/')[-1]+'. No EXIF data.')
                errorlog.write(fname.split('/')[-1]+' no EXIF data')
                continue
        try:
            date = datetime.strptime(exif['DateTime'], '%Y:%m:%d %H:%M:%S')
        except:
            print('Problem with ' + fname.split('/')[-1] + '. Bad date format.')
            errorlog.write(fname.split('/')[-1]+' bad EXIF date format')
            continue
        newfname=imgdir+date.strftime("%Y-%m-%d_%H.%M.%S")+extension
        os.rename(fname,newfname)
    errorlog.close()
    print('Photos renamed successfully!')

if __name__ == "__main__":
    # Modifications to config formats
    reformat = {}
    reformat['addslash'] = ['imgdir']
    #Read in config file
    config =  util.readConfig('./config/renamePhotos_config.txt',reformat=reformat)

    main(**config)