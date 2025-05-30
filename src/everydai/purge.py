from datetime import datetime, timedelta
import shutil
from pathlib import Path

import numpy as np

import src.everydai.utils.utils as util
import src.everydai.utils.utils_config as util_config


class Purger:

    def __init__(self):
        config = util_config.read_config('./config.txt')
        self.config_main = config['main']
        self.config_dir = util.dir_slash(config['directories'])
        self.config_review = config['review']

        self.fnames_main, self.dates_main = util.image_finder(config['main'], self.config_dir['imgdir'])
        _, self.dates_review = util.image_finder(config['main'], self.config_dir['reviewdir'])

    def run(self):
        # Call the main function with the instance variables
        self.purge()

    def purge(self):

        # TODO: validate config values
        # if datefinish < datestart:
        #     print('ERROR: end date (datefinish) is earlier than the start date (datestart).')
        #     sys.exit()

        # Initialise parameters
        datestart = datetime.strptime(self.config_main['datestart'], '%Y-%m-%d')
        datefinish = datetime.strptime(self.config_main['datefinish'], '%Y-%m-%d')
        days = (datefinish-datestart).days
        # Construct dates running from the start date
        testdate1 = datestart.replace(hour=int(self.config_review["sleepstart"]))
        testdate2 = datestart.replace(hour=int(self.config_review["sleepfinish"])) + timedelta(days=1)
        for i in range(days):
            # Find all images taken on that day
            imgdates = self.dates_main[np.logical_and(self.dates_main > testdate1, self.dates_main < testdate2)]
            # Check if there's only one photo left for this day in the review directory
            reviewcheck = self.dates_review[(self.dates_review > testdate1) & (self.dates_review < testdate2)]
            if reviewcheck.size != 1:
                if imgdates.size == 1 and reviewcheck.size != 0:
                    # Extra clause - already reviewed and purged, but some review photos remain
                    print(f"Photos from {testdate1.strftime('%Y-%m-%d_%H.%M.%S')}"
                          " to {testdate2.strftime('%Y-%m-%d_%H.%M.%S')}"
                          " are already reviewed, but review photos remain. Please remove them manually.")
                elif reviewcheck.size > 1:
                    print(f"Photos from {testdate1.strftime('%Y-%m-%d_%H.%M.%S')}"
                          " to {testdate2.strftime('%Y-%m-%d_%H.%M.%S')}"
                          " still need reviewing.")
                elif reviewcheck.size == 0 and imgdates.size > 1:
                    print(f"No photos left from {testdate1.strftime('%Y-%m-%d_%H.%M.%S')}"
                          " to {testdate2.strftime('%Y-%m-%d_%H.%M.%S')}"
                          " in the review directory, but the day still needs reviewing.")
                testdate1 += timedelta(days=1)
                testdate2 += timedelta(days=1)
                continue

            # Purge image photos ONLY IF there's more than one on that day
            if imgdates.size > 1:
                for imgdate in imgdates:
                    index = np.argwhere(self.dates_main == imgdate)[0][0]
                    fname = self.fnames_main[index]
                    sname = Path(self.config_dir["purgedir"]) / fname.name
                    if imgdate not in self.dates_review:
                        print('Purging ' + fname.name)
                        shutil.move(fname, sname)
                        pass
                    else:
                        rname = Path(self.config_dir["reviewdir"]) / (fname.stem +
                                                                      '_review' + self.config_main["extension"])
                        deletname = Path("delet_review") / rname.name  # TEMPORARY
                    #     os.remove(rname)
                        shutil.move(rname, deletname)  # TEMPORARY
            testdate1 += timedelta(days=1)
            testdate2 += timedelta(days=1)
        print('Purging completed successfully!')


if __name__ == "__main__":
    # Create a Purger instance and run the review process
    reviewer = Purger()
    reviewer.run()
