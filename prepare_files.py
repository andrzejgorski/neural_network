from PIL import Image
import numpy as np
import os


class FilePreparer(object):
    def __init__(self, path, output_path):
        self.basewidth = 64
        self.counter = 0
        self.known_dirs = []
        self.path = path
        self.output_path = output_path

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
                    output_img = np.array(
                        list(pil_imgray.getdata(band=0)), float)

                    file_output_path = os.path.join(
                        self.output_path,
                        '/{:04d}.in'.format(self.counter)
                    )

                    with open(file_output_path, 'w') as f:
                        f.write(
                            ' '.join([str(p/255) for p in output_img]) + '\n'
                        )
                        f.write(self.get_output(directory))

                    self.counter += 1

    def prepare_files(self):
        if not (os.path.isfile(self.path)):
            print('File is empty skipping step')
        else:
            print('Starting to prepare files.')
            for subdir, dirs, files in os.walk(self.path):
                for directory in dirs:
                    self.check_dir(subdir, directory)
            print('Ended preparing files.')


if __name__ == "__main__":
    fp = FilePreparer()
    fp.prepare_files()
