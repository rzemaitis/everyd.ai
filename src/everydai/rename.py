from datetime import datetime
from pathlib import Path

from PIL import Image, ExifTags

import everydai.utils.utils_config as util_config


class Renamer:

    def __init__(self):
        config = util_config.read_config('./config.txt')
        self.config_main = config['main']
        self.config_dir = config['directories']

    def image_finder(self):
        """
        Find all images with the given extension.
        This is custom made for the Renamer class.
        """
        path = Path(self.config_dir["imgdir"])
        fnames = list(path.glob(f"*{self.config_main['extension']}"))

        if len(fnames) == 0:
            raise FileNotFoundError(
                f"No images found in {self.config_dir['imgdir']} "
                f"with extension {self.config_main['extension']}. "
            )
        return fnames

    def rename(self):
        fnames = self.image_finder()
        for fname in fnames:
            with Image.open(fname) as img:
                # Taken from https://stackoverflow.com/questions/21697645/
                # how-to-extract-metadata-from-a-image-using-python
                try:
                    exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}
                except AttributeError:
                    print(f"Problem with {fname.name}. No EXIF data.")
                    continue
            try:
                date = datetime.strptime(exif['DateTimeOriginal'], '%Y:%m:%d %H:%M:%S')
            except ValueError:
                print(f"Problem with {fname.name}. Bad date format.")
                continue
            newfname = Path(self.config_dir['imgdir']) / (f"{date.strftime('%Y-%m-%d_%H.%M.%S')}"
                                                          f"{self.config_main['extension']}")
        try:
            Path(fname).rename(newfname)
        except FileExistsError:
            print(f"File {newfname.name} already exists. Duplicate or photo taken at the same second.")
        except OSError as e:
            print(f"Problem renaming {Path(fname).name}: {e}")
        print('Photos renamed successfully!')


if __name__ == "__main__":
    renamer = Renamer()
    renamer.rename()
