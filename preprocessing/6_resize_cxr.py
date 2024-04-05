import argparse
from PIL import Image
from PIL import ImageFile
import glob
from multiprocessing.dummy import Pool
from tqdm import tqdm
import os

# goal: resize the shorter edge of input images to 384 and limit the longer edge to under 640
# while preserving the aspect ratio

parser = argparse.ArgumentParser(description="6_resize_cxr.py")
parser.add_argument('input_path', type=str, help="'cxr' directory in 'scratch' where all MIMIC CXR images are stored.")
parser.add_argument('output_path', type=str, help="'cxr_dataset' directory where all resized MIMIC CXR images should be stored.")
args = parser.parse_args()

ImageFile.LOAD_TRUNCATED_IMAGES = True

paths_all = glob.glob(f'{args.input_path}/**/*.jpg', recursive = True)  # file paths of all CXR images in 'scratch/cxr'
print('CXR num total:', len(paths_all))

paths_resized = glob.glob(f'{args.output_path}/**/*.jpg', recursive = True) # file paths of resized CXR images
print('CXR num resized:', len(paths_resized))

resized_files = [os.path.basename(path) for path in paths_resized]  # filenames of resized CXR images
paths = [path for path in paths_all if os.path.basename(path) not in resized_files] # file paths of CXR images to be resized
print('CXR num to be resized:', len(paths))

def resize_images(args, path):
    # extract the filename and path in order to save the resized image using the same directory structure
    filename = path.split('/')[-1]  # dicom_id.jpg
    filepath = path.split('/')[-4] + '/' + path.split('/')[-3] + '/' + path.split('/')[-2]  # pXX/pXXXXXXXX/sYYYYYYYY

    img = Image.open(path)
    w, h = img.size
    if w < h:
        w_new = 384
        h_new = int(float(h)*float(w_new/w))
        if h_new > 640:
            h_new = 640
            w_new = int(float(w)*float(h_new/h))
    else:
        h_new = 384
        w_new = int(float(w)*float(h_new/h))
        if w_new > 640:
            w_new = 640
            h_new = int(float(h)*float(w_new/w))
    img = img.resize((w_new,h_new))
    
    # create required directories and save the resized images
    output_dir = f'{args.output_path}/{filepath}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    img.save(f'{output_dir}/{filename}')

# iterate through 10 images at a time and resize them (i.e. 10 images per thread)
threads = 10
for i in tqdm(range(0, len(paths), threads)):
    paths_subset = paths[i : i+threads]
    pool = Pool(len(paths_subset))
    pool.map(lambda path: resize_images(args, path), paths_subset)
    pool.close()
    pool.join()