from PIL import Image
import numpy as np
import os


class FilePreparer(object):
    def __init__(self):
        self.basewidth = 64
        self.counter = 0
        self.known_dirs = []

    def get_output(self, directory):
        if directory not in self.known_dirs:
            self.known_dirs.append(directory)
        result = (['0'] * 62)
        result[self.known_dirs.index(directory)] = '1'
        return ' '.join(result)

    def check_dir(self, subdir, directory):
        for subdir, dirs, files in os.walk(os.path.join(subdir, directory)):
            for file_ in files:
                if file_.endswith('.ppm'):
                    img = Image.open(os.path.join(subdir, file_))
                    resized_img = img.resize(
                        (self.basewidth, self.basewidth), Image.ANTIALIAS)
                    pil_imgray = resized_img.convert('LA')
                    output_img = np.array(list(pil_imgray.getdata(band=0)), float)
                    with open('inputs/{:04d}.in'.format(self.counter), 'w') as f:
                        f.write(' '.join([str(p/255) for p in output_img]) + '\n')
                        f.write(self.get_output(directory))

                    self.counter += 1

    def prepare_files(self):
        for subdir, dirs, files in os.walk('Training'):
            for directory in dirs:
                self.check_dir(subdir, directory)


fp = FilePreparer()
fp.prepare_files()
