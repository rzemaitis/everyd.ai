import argparse
import everydai.utils.utils_config as utils_config
import everydai.utils.utils as util


def main(args):
    utils_config.create_config('./')
    util.create_dir('Images')
    util.create_dir('Movie')
    util.create_dir('Review')
    util.create_dir('Purged')
    util.create_dir('Solutions')

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use prepareWorkspace.py to create '
        '(or reinitialise) config files.'
        )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
