import argparse
import numpy as np
from pathlib import Path
import pickle

import everydai.utils.utils as util
import everydai.utils.utils_config as utils_config


class Runner:

    def __init__(self, args):
        # Read in config file
        config = utils_config.read_config('./config.txt')
        self.config_main = config['main']
        # TODO: get rid of dir slash. Path will sort it.
        self.config_dir = util.dir_slash(config['directories'])
        self.sleepstart = int(config['review']['sleepstart'])
        self.cores = int(config['multiprocessing']['cores'])

        # TODO: more functionality beyond template?
        template = config['dimensions']['template']
        self.dim = util.read_image(template).shape[:2]

        self.fnames, _ = util.image_finder(self.config_main, self.config_dir['imgdir'], sleepstart=self.sleepstart)
        # Cross-match the solutions with images
        self.match_solutions()

    def match_solutions(self):
        """
        Clean the filenames and solution name lists keeping only the ones that match.
        TODO: update docstring.
        """
        # # Find all solution files
        # solnames = list(Path(self.config_dir['solutionsdir']).glob('*.pickle'))
        # # Extract solution dates
        # # TODO: match the sleepstart dates...
        # solpath = os.path.dirname(solnames[0]).replace(os.sep, '/')
        # soldatestrs = [os.path.basename(fname).split('.')[0] for fname in solnames]
        # fnames_culled, solnames_culled = [], []
        # for fname in fnames:
        #     # Extract image date
        #     datestr = util.fname_to_date(fname)
        #     datestr =
        #     if datestr in soldatestrs:
        #         fnames_culled.append(fname)
        #         solnames_culled.append(f"{solpath}/{datestr}.pickle")
        #     print(datestr)
        # # TODO: raise error if no solnames matched
        # # print("fnames", fnames)
        # # print("solnames",solnames_culled)
        # return fnames_culled, solnames_culled

        # Extract the date strings for each image and solution
        dates = util.daily_dates(self.fnames, sleepstart=self.sleepstart)
        gen = Path(self.config_dir['solutionsdir']).glob('*.pickle')
        dates_sol = np.array([p.stem for p in gen])
        # Find common dates
        common_dates = np.intersect1d(dates, dates_sol)
        # TODO: raise error if no dates matched

        # Cull solutions
        datemask = np.ones(dates_sol.size, dtype=bool)
        for i, date_sol in enumerate(dates_sol):
            if date_sol not in common_dates:
                datemask[i] = False
        self.dates_sol = np.array(dates_sol)[datemask]

        # Cull fnames
        datemask = np.ones(dates.size, dtype=bool)
        for i, date in enumerate(dates):
            if date not in common_dates:
                datemask[i] = False
        self.fnames = np.array(self.fnames)[datemask]

    def transform_image(self, fname, params):
        """
        Transform the image instance with the given parameters:
        - translate
        - scale
        - rotate
        """

        # Read in parameters
        xoff, yoff, scale, angle = params

        # Read in the image
        img = util.read_image(fname)

        # Rescale to template size
        img = util.rescale(img, self.dim)

        # Translate image
        img = util.translate_image(img, xoff, yoff)

        # Scale image
        img = util.cv2_clipped_zoom(img, scale)

        # Rotate image
        img = util.rotate_image(img, angle)

        return img

    def save_image(self, fname, img, add_date=False):

        # Add _movie to the filename
        stem = Path(fname).stem.rpartition('_')[0]  # Get the part before the last underscore
        sname = Path(self.config_dir['moviedir']) / f"{stem}_movie{self.config_main['extension']}"

        # Add a date to the image in the corner
        if add_date:
            img = util.image_add_date(img, stem)

        print(f"Image saved: {sname}")
        util.write_image(sname, img)

    def image_creation_runner(self):

        # TODO: make an error log saying which solutions are missing instead
        # if len(baddates > 0):
        #     print('No solutions found for dates', baddates)

        # Choose filenames by date
        # if usedate:
        #     dates = util.fname_to_date(fnames, dateformat = "%Y-%m-%d")
        #     fnames = [fnames[i] for i in np.where((dates > datestart) &
        #               (dates < datefinish))[0]]

        # In the case of no solutions found
        # if len(fnames) == 0:
        #     print(f"No solutions found in {config_dir['solutionsdir']}.")
        #     if config_main['datestart'] != '' and
        #            config_main['datefinish'] != '':
        #         print('Try changing your selected date range.')
        #     sys.exit()

        # Run saving in a for loop
        for fname, date_sol in zip(self.fnames, self.dates_sol):
            # Read in transformation from the solution file"
            path_solution = Path(self.config_dir['solutionsdir']) / f"{date_sol}.pickle"
            params = pickle.load(open(path_solution, 'rb'))
            # Transform using the solved parameters
            image = self.transform_image(fname, params)
            # Save the image
            self.save_image(fname, image, add_date=True)
        print("Images created successfully!")

    # def image_solution_runner(self):
    #     start = time.time()
    #     # Read in and reshape template image if needed
    #     # img_template = util.read_image(self.config_dim['template'])

    #     # alignpartial = partial(align, img_template, **kwargs)
    #     if self.cores == 1:
    #         aligner = Aligner(self.config_dim["template"])
    #         for fname in self.fnames:
    #             aligner.align(fname)
    #     # else:
    #     #     with Pool(self.cores) as pool:
    #     #     # pool = Pool(processes=cores)
    #     #         # TODO: Figure out why fnames are chosen at random
    #     #         # TODO: Do the fitting first and save in files,
    #     #         # then do the saving separately
    #     #         pool.map(alignpartial, fnames)
    #     #         pool.close()
    #     #         pool.join()

    #     # errorlog.close()
    #     print('Aligning finished successfully!')
    #     end = time.time()
    #     print('Seconds spent:', end - start)

    # def image_finder(self):
    #     # Find all images in the directory
    #     fnames = glob.glob(f"{self.config_dir['imgdir']}*{self.config_main['extension']}")

    #     # If a specific date range is supplied, cull by date
    #     datemask = np.ones(len(fnames), dtype=bool)
    #     dates = util.fname_to_date(fnames)
    #     if self.config_main['datestart'] != '':
    #         datestart = datetime.strptime(self.config_main['datestart'], '%Y-%m-%d')
    #         datemask = datemask & (dates > datestart)
    #     if self.config_main['datefinish'] != '':
    #         datefinish = datetime.strptime(self.config_main['datefinish'], '%Y-%m-%d'
    #                                        ) + timedelta(days=1)
    #         datemask = datemask & (dates < datefinish)

    #     # Choose picture names by date if we need to
    #     if not all(datemask):
    #         fnames = [fnames[i] for i in np.where(datemask)[0]]

    #     # In the case of no images found
    #     if len(fnames) == 0:
    #         raise FileNotFoundError(
    #             f"No images found in {self.config_dir['imgdir']} "
    #             f"with extension {self.config_main['extension']}. "
    #             "Try changing your selected date range."
    #         )
    #     return fnames

    def run(self):

        # if args.create_images:
        #     image_creation_runner(config_dir, fnames)
        # else:
        self.image_creation_runner()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use align.py to either create aligning solutions or '
                    'aligned images with premade solutions.'
        )
    # TODO: actually use the arguments
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite already existing solutions or images.')
    return parser.parse_args()


def main(args):
    aligner = Runner(args)
    aligner.run()


if __name__ == "__main__":
    main(parse_args())
