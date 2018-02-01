import os
import os
import shutil
import glob
from findROIImage import extrct_ROIImage

# for moving pgm files
path_to_current_dir = "/home/ahtesham//"
path_to_ddsm = "../../DDSM/cases/cancers/"
for root, subFolders, file_names in os.walk(path_to_current_dir):
    for file_name in file_names:

        # if ".ics" in file_name:
            # print root
            # dirname = os.path.basename(root)
            # ics = os.path.join(root, file_name)
            # #shutil.copyfile('../aatif/code/template/case_make_template' , root+'/case_make_template')
            # #shutil.copyfile('../aatif/code/template/mktemplate', root + '/mktemplate')
            # cmd =  '~/aatif/code/template/aq/'+ dirname +'/case_make_template {0}  {1}'.format('~/aatif/code/template/aq/'+dirname+'/'+file_name , dirname)
            # os.system(cmd)
            # os.remove("ChangedFile.csv")
        # print file_name
        if ".LJPEG.1.template.pgm" in file_name:
            in_pgm_file_path = os.path.join(root, file_name)
            fileInfo = file_name.split('.')
            case_dir = fileInfo[0].split('_')[1];
            out_pgm_file_path = os.path.join(path_to_ddsm + "case"+case_dir , file_name)
            print out_pgm_file_path;
            shutil.copyfile(in_pgm_file_path ,out_pgm_file_path);
            os.remove(in_pgm_file_path)
print "done moving pgm files"