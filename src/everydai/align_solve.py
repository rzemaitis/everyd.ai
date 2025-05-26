import argparse
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import pickle
import time

import everydai.utils.utils as util
import everydai.utils.utils_config as utils_config


class Aligner:
    """
    Class to align images using a template image.

    Steps taken:
    1. Read in the template image and find the face points.
    2. Read in the image and find the face points.
    3. Align the image to the template image using the face points.
    4. Return solution parameters for the alignment.
    """

    def __init__(self, fname_template, solutions_dir='./Solutions'):
        self.dim, self.dim_original, self.dim_template = np.zeros(2), np.zeros(2), np.zeros(2)
        self.points, self.points_template = np.zeros(68), np.zeros(68)
        self.size_ratio = 1
        self.solutions_dir = solutions_dir
        self.face_detectors = util.init_face_detectors()

        # Read in the template image
        self.read_template(fname_template)

    def update_image(self, fname):
        """
        Read in the image
        Update the relevant parameters
        """
        self.read_target_image(fname)
        self.size_ratio = np.hypot(*self.dim) / np.hypot(*self.dim_original)
        # Default values
        # Center of the image, scaled by image diagonal, no rotation
        self.default_guess = np.array([0.0, 0.0, np.hypot(*self.dim) /
                                       np.hypot(*self.dim_original), 0.])
        # Bound within the image dimensions, no more than 3x scale, no more than 90 degrees rotation
        self.default_bounds = [(-self.dim[1], self.dim[1]), (-self.dim[0], self.dim[0]),
                               (self.size_ratio / 5, self.size_ratio * 5), (-90, 90)]

    def read_template(self, fname):
        """
        Read in the template image
        Save the dimensions and face points
        The image is not kept
        """
        img_template = util.read_image(fname)
        self.dim_template = img_template.shape[:2]
        self.points_template = util.detect_face(img_template, *self.face_detectors)
        # Failsafe if no faces were found
        # TODO: raise an error?
        # if self.points is None:
        #     return None

    def read_target_image(self, fname):
        """
        Read in the image to be aligned
        Rescale it to the template dimensions
        Save the dimensions, image name and and face points
        """
        # Read in the image
        img = util.read_image(fname)
        # Find the xy dimensions of the image
        # These are the "original" dimensions of the image before scaling.
        self.dim_original = img.shape[:2]
        img = util.rescale(img, self.dim_template)
        self.dim = img.shape[:2]

        self.points = util.detect_face(img, *self.face_detectors)

    # def nose_rotate(self, img):
    #     # TODO: make a nose vector, rotate the image to make the vector angle zero.
    #     # Detect face
    #     # Use top and bottom of nose landmarks
    #     # Make a vector (choose what direction - top/bottom)
    #     # Make a zero  vector pointing straight down
    #     # Calculate angle
    #     # rotate by negative angle (?)
    #     return img

    # def eye_fitter(self, guess=None, bounds=None):
    #     '''Fit eyes only'''
    #     # Null-hypothesis guess - x_c, y_c, scale, rotation
    #     if guess is None:
    #         guess = copy(self.default_guess)
    #     if bounds is None:
    #         bounds = copy(self.default_bounds)

    def bound_rescaler(self, scale):
        """
        Rescale bounds for finer solution searches.
        """

    def solver(self):
        """
        Find the solution for image alignment.
        Step 1: Fit the eyes first.
        Step 2: Fit the whole face.
        """
        # TODO: Rotation is too volatile. Need to fit the other parameters first.
        # TODO: the nose might potentially solve this issue, if the eyes don't.
        # Step 1: Fit eyes first - they eyes start from 36 to the end of the array
        res = minimize(util.costfunction, self.default_guess,
                       args=(self.points[36:48], self.points_template[36:48], self.dim),
                       bounds=self.default_bounds)
        guess = res.x
        # Step 2: Include the whole face
        scale = 0.2
        refined_bounds = [(x * scale, y * scale) for (x, y) in self.default_bounds]
        res = minimize(util.costfunction, guess,
                       args=(self.points, self.points_template, self.dim), bounds=refined_bounds)
        guess = res.x
        return guess

    def align(self, fname, date, solnsdir='./Solutions', sleepstart=0):
        # TODO: move the solnsdir to the config file readout
        self.update_image(fname)
        solution = self.solver()
        # # Check
        # if params is None:
        #     print('No face found in file ' + basename)
        #     errorlog = open('badphotos_align.txt', "a")
        #     errorlog.write('No face found in file ' + basename)
        #     errorlog.close()
        #     return None

        # Save the parameters
        # TODO: move the Path to the config file readout
        sname = Path(solnsdir) / f"{date}.pickle"
        pickle.dump(solution, open(sname, "wb"))
        print(f"{sname} finished.")


class Runner:

    def __init__(self, args):
        # Read in config file
        config = utils_config.read_config('./config.txt')
        self.config_main = config['main']
        self.config_dim = config['dimensions']
        # TODO: util.dir_slash is obsolete with Path.
        self.config_dir = util.dir_slash(config['directories'])
        self.sleepstart = int(config['review']['sleepstart'])
        self.cores = int(config['multiprocessing']['cores'])
        self.fnames, _ = util.image_finder(self.config_main, self.config_dir['imgdir'],
                                           sleepstart=self.sleepstart)
        self.dates = util.daily_dates(self.fnames, self.sleepstart)

    # TODO: Maybe useful for the transformer class
    # def image_creation_runner(config_dir, fnames):
    #     # Check which solutions exist
    #     solnames = glob.glob(f"{config_dir['solutionsdir']}*.pickle")
    #     if len(fnames) == 0:
    #         print('No solutions found.')
    #         print('Try changing your selected date range.')
    #         sys.exit()

        # # Cross-match the solutions with selected images
        # solnames, baddates = match_solutions(fnames, solnames)
        # # TODO: make an error log saying which solutions are missing instead
        # if len(baddates > 0):
        #     print('No solutions found for dates', baddates)

        # # Choose filenames by date
        # if usedate:
        #     dates = util.fname_to_date(fnames, dateformat = "%Y-%m-%d")
        #     fnames = [fnames[i] for i in np.where((dates > datestart) &
        #               (dates < datefinish))[0]]

        # # In the case of no solutions found
        # if len(fnames) == 0:
        #     print(f"No solutions found in {config_dir['solutionsdir']}.")
        #     if config_main['datestart'] != '' and
        #            config_main['datefinish'] != '':
        #         print('Try changing your selected date range.')
        #     sys.exit()

        # Run saving in a for loop
        # for fname in fnames:
        #     if fname in solnames.keys():
        #         solname = solnames[fname]
        #     params = pickle.load(open(fname,'rb'))
        #     #Transform using the solved parameters

        # Read in transformation from
        # TODO: maybe we don't even need to return the image here?
        # img = transform_image(img, params)
        # save_image(img, sname, params)
        # pass

    def image_solution_runner(self):
        start = time.time()
        # TODO: figure out dimensions, if we even need them
        # Read in and reshape template image if needed
        # img_template = util.read_image(self.config_dim['template'])

        # Initialise dimensions
        # TODO: you'll have to pass this as a target dimension to the aligner class
        # # think about it later
        # if self.config_dim['dimfile'] != '':
        #     img_dim = util.read_image(self.config_dim['dimfile'])
        #     dim = img_dim.shape[:2]
        # # Supplied dimensions
        # elif self.config_dim['height'] != '' and self.config_dim['width'] != '':
        #     dim = (int(self.config_dim['height']), int(self.config_dim['width']))
        # # No dimensions supplied: use the template image
        # else:
        #     dim = img_template.shape[:2]

        # If the required dimensions are not the same as
        # template file's dimensions,
        # rescale the template image
        # TODO: same as above, think about it later
        # if dim != img_template.shape[:2]:
        #     img_template = util.rescale(img_template, dim)

        # Initialise other things for aligning
        # points_template = util.find_face(img_template)
        # kwargs = {'solnsdir': self.config_dir['solutionsdir'],
        #             'points_template': points_template}

        # #Initialise error log as a text file
        # errorlog = open('badphotos_align.txt', "w")
        # errorlog.close()

        # alignpartial = partial(align, img_template, **kwargs)
        if self.cores == 1:
            aligner = Aligner(self.config_dim["template"])
            for fname, date in zip(self.fnames, self.dates):
                aligner.align(fname, date, sleepstart=self.sleepstart)
        # else:
        #     with Pool(self.cores) as pool:
        #     # pool = Pool(processes=cores)
        #         # TODO: Figure out why fnames are chosen at random
        #         # TODO: Do the fitting first and save in files,
        #         # then do the saving separately
        #         pool.map(alignpartial, fnames)
        #         pool.close()
        #         pool.join()

        # errorlog.close()
        print('Aligning finished successfully!')
        end = time.time()
        print('Seconds spent:', end - start)

    def run(self):
        self.image_solution_runner()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use align.py to either create aligning solutions or '
                    'aligned images with premade solutions.'
        )
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite already existing solutions or images.')

    parser.add_argument('--create_images', action='store_true',
                        help='Create images with the solutions found.')
    return parser.parse_args()


def main(args):
    aligner = Runner(args)
    aligner.run()


if __name__ == "__main__":
    main(parse_args())
