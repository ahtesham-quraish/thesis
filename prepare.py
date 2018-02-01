import os
import shutil
import glob
from findROIImage import extrct_ROIImage
path_to_ddsm = "/home/ahtesham/train/"

template_dir = "../aatif/code/template/mktemplate"
index = 1 ;
dindex = 1
cat_train_path = '/home/ahtesham/catanddog/train/cat/'
dog_train_path = '/home/ahtesham/catanddog/train/dog/'
cat_test_path = '/home/ahtesham/catanddog/test/cat/'
dog_test_path = '/home/ahtesham/catanddog/test/dog/'
cat_validation_path = '/home/ahtesham/catanddog/validation/cat/'
dog_validation_path = '/home/ahtesham/catanddog/validation/dog/'
for root, subFolders, file_names in os.walk(path_to_ddsm):
    for file_name in file_names:
        if 'cat' in file_name:
            if index > 1000 and index <2500:
                file_path = os.path.join(root, file_name);
                out_path = os.path.join(cat_test_path, file_name)
                shutil.copyfile(file_path, out_path)
            if index > 2500 and index < 4000:
                file_path = os.path.join(root, file_name);
                out_path = os.path.join(cat_validation_path, file_name)
                shutil.copyfile(file_path, out_path)
            index = index +1;
        if 'dog' in file_name:
            if dindex > 1000 and dindex < 2500:
                file_path = os.path.join(root, file_name);
                out_path = os.path.join(dog_test_path, file_name)
                shutil.copyfile(file_path, out_path)
            if dindex > 2500 and dindex < 4000:
                file_path = os.path.join(root, file_name);
                out_path = os.path.join(dog_validation_path, file_name)
                shutil.copyfile(file_path, out_path)
            dindex = dindex + 1