import os
import os
import shutil
import glob
from findROIImage import extrct_ROIImage




path_patch_dataset = "/home/ahtesham/patch_dataset/"
path_to_ddsm = "../patchset/"
index = 0
template_dir = "../../aatif/code/template/mktemplate"
for root, subFolders, file_names in os.walk(path_to_ddsm):
    for file_name in file_names:

        if ".jpg" in file_name:
            index = index +1 ;
            print file_name , index;
            out_path = os.path.join(root, file_name)
            os.remove(out_path)
            dirname = os.path.basename(root)



print('done generating pgm files')

