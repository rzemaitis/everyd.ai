from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

import everydai.utils.utils as util
import everydai.utils.utils_config as util_config


class Reviewer:
    def __init__(self):
        config = util_config.read_config('./config.txt')
        self.config_main = config['main']
        self.config_dir = util.dir_slash(config['directories'])
        self.config_review = config['review']

        self.fnames, self.dates = util.image_finder(config['main'], config['directories']['imgdir'])
        self.dates = np.array(self.dates)

    def run(self):
        # Call the main function with the instance variables
        self.prepare_review()

    def prepare_review(self, puttemplate=False):
        # Initialise boolean to check if we need to store photo names taken at unusual time
        # needlog = False

        # Initialise dates and edit them according to sleep time
        sleepstart = int(self.config_review['sleepstart'])
        sleepfinish = int(self.config_review['sleepfinish'])
        datestart = datetime.strptime(self.config_main['datestart'],
                                      '%Y-%m-%d').replace(hour=sleepstart)
        datefinish = datetime.strptime(self.config_main['datefinish'],
                                       '%Y-%m-%d').replace(hour=sleepfinish) + timedelta(days=1)

        # TODO: Better error, or overall config validation
        # if datefinish < datestart:
        #     print('FATAL ERROR: end date (datefinish) is earlier than the start date (datestart).')
        #     sys.exit()

        # Initialise parameters
        # dim = (0, 0)
        days = (datefinish - datestart).days
        # Construct dates running from the start date
        testdate1 = datestart
        testdate2 = datestart + timedelta(days=1)
        needsreview = 0  # Counter to check how many dates in the given ranges have not been reviewed yet
        for i in range(days):
            # Find suitable days
            gooddates = self.dates[np.logical_and(self.dates > testdate1, self.dates < testdate2)]
            # Put photos up for review ONLY IF there's more than one on that day
            if gooddates.size > 1:
                # Change flag that checks if all dates have only one picture
                needsreview += 1
                for gooddate in gooddates:
                    index = np.argwhere(self.dates == gooddate)[0][0]
                    fname = self.fnames[index]
                    img = util.read_image(fname)
                    sname = Path(self.config_dir['reviewdir']) / (fname.stem + '_review' + fname.suffix)
                    # Put this picture to log to check why it's taken at a weird time
                    if sleepstart < gooddate.hour < sleepfinish:
                        print('Unusual photo time found in', testdate1.strftime('%Y-%m-%d'))
                        # if not needlog:
                        #     needlog = True
                        #     log = open(reviewdir + 'unusualtime.txt', "w")
                        # log.write(fname.split('/')[-1] + '\n')

                    # # Put a template on top of image and copy it
                    # if puttemplate:
                    #     # If dimensions are the same as the last picture
                    #     if dim == img.shape[:2]:
                    #         pass
                    #     else:
                    #         dim = img.shape[:2]
                    #         try:
                    #             points = np.loadtxt(Path(reviewdir) / f"template_{dim[1]}{dim[0]}.txt")
                    #         except FileNotFoundError as e:
                    #             print('ERROR: '+fname)
                    #             print('File template_' + str(dim[1]) + 'x' +
                    # str(dim[0]) + '.txt not found in ' + self.config_dir['reviewdir'])
                    #             print('Please set puttemplate=False or run this first:')
                    #             print('templateMaker.py <templateimage(full path)> <reviewdirectory> width=' + str(
                    #                 dim[1]) + ' height=' + str(dim[0]))
                    #             raise e
                    #     for p in points:
                    #         x, y = p.astype(int)
                    #         circlesize = int(3 * np.hypot(dim[0], dim[1]) / 1200)  # Scaled by the diagonal
                    #         cv2.circle(img=img, center=(x, y), radius=circlesize, color=(0, 255, 0),
                    #                 thickness=-1)
                    img = util.image_add_date(img, testdate1.strftime('%Y-%m-%d'))
                    print("Saving image to", sname)
                    util.write_image(sname, img)
            testdate1 += timedelta(days=1)
            testdate2 += timedelta(days=1)
        if needsreview == 0:
            print(f"All dates have already been reviewed from {datestart.strftime('%Y-%m-%d')}",
                  f" to {(datefinish - timedelta(days=1)).strftime('%Y-%m-%d')}.")
        else:
            print(f"{needsreview} days need reviewing from {datestart.strftime('%Y-%m-%d')}"
                  f" to {(datefinish - timedelta(days=1)).strftime('%Y-%m-%d')}.")
        print('Review photos created successfully!')


def main():
    # Create a Reviewer instance and run the review process
    reviewer = Reviewer()
    reviewer.run()


if __name__ == "__main__":
    main()
