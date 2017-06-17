import os
import numpy
import pickle


def get_meta(input_dir='mpii-128', output_file='mpii_meta.pkl'):
    meta = {}
    cnt = 0
    image_dirs = os.listdir(input_dir)
    for image_dir in image_dirs:
        sub_dirs = os.listdir(os.path.join(input_dir, image_dir))
        sub_dirs.sort()
        for sub_dir in sub_dirs:
            files = os.listdir(os.path.join(input_dir, image_dir, sub_dir))
            files.sort()
            meta[cnt] = [image_dir, sub_dir, files]
            cnt += 1
    pickle.dump(meta, open(output_file, 'w'))


def main():
    get_meta()

if __name__ == '__main__':
    main()

