import os
import os
import shutil
import glob
from findROIImage import extrct_ROIImage

# for moving pgm files
path_to_current_dir = "/home/ahtesham/thesis/pgm_files_bn/"
path_to_ddsm = "../../DDSM/cases/benigns/"
index = 0
for root, subFolders, file_names in os.walk(path_to_current_dir):
    for file_name in file_names:
        if ".LJPEG.1.template.pgm" in file_name:
            index = index + 1
            in_pgm_file_path = os.path.join(root, file_name)
            fileInfo = file_name.split('.')
            case_dir = fileInfo[0].split('_')[1];
            for outerroot, subFolders, file_names in os.walk(path_to_ddsm):
                for file_name_in in file_names:
                  try:
                    if ".ics" in file_name_in:
                        if case_dir in outerroot:
                            out_pgm_file_path = os.path.join(outerroot , file_name)
                            shutil.copyfile(in_pgm_file_path ,out_pgm_file_path);
                            os.remove(in_pgm_file_path)
                  except Exception:
                      print out_pgm_file_path, index, in_pgm_file_path
                      print '\n'
                      print(case_dir)
                      pass
            #out_pgm_file_path = os.path.join(path_to_ddsm + "case"+case_dir , file_name)
            #print out_pgm_file_path;
            #shutil.copyfile(in_pgm_file_path ,out_pgm_file_path);
            #os.remove(in_pgm_file_path)
print "done moving pgm files"