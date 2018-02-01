import os
import os
import shutil
import glob
from findROIImage import extrct_ROIImage



path_patch_dataset = "/home/ahtesham/patch_dataset/"
path_to_ddsm = "../../DDSM/cases/cancers/"

template_dir = "../../aatif/code/template/mktemplate"
for root, subFolders, file_names in os.walk(path_to_ddsm):
    for file_name in file_names:
#         print "";
        if ".ics" in file_name:
            print root
            dirname = os.path.basename(root)
            ics = os.path.join(root, file_name)
            shutil.copyfile('../../aatif/code/template/case_make_template' , root+'/case_make_template')
            shutil.copyfile('../../aatif/code/template/mktemplate', root + '/mktemplate')
            os.chmod(root + '/mktemplate', 777)
            os.chmod(root+'/case_make_template', 777)
            cmd =  root+'/case_make_template {0}  {1}'.format(root+'/'+file_name , dirname)
            os.system(cmd)


print('done generating pgm files')

