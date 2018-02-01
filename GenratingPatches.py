import os
import os
import shutil
import glob
import csv
from findROIImage import extrct_ROIImage

path_to_ddsm = "../DDSM/cases/"
path_patch_dataset = '/home/ahtesham/patchset/'
index = 0

with open('/home/ahtesham/patchset/record.csv', 'w') as csvfile:
    fieldnames = ['root', 'file', 'lesion_type', 'PATHOLOGY']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for root, subFolders, file_names in os.walk(path_to_ddsm):
        for file_name in file_names:
            try:

                 if '.LJPEG.1.template.pgm' in file_name :

                      index = index + 1;
                      if index < 20000:
                          #print(file_name , root)
                          fileInfo = file_name.split('.')
                          jpeg_file_name = fileInfo[0]+'.'+fileInfo[1]+'.jpg';
                          overlay_file_name = fileInfo[0] + '.' + fileInfo[1] + '.OVERLAY';
                          pgm_file_path = os.path.join(root, file_name)
                          jpeg_file_path = os.path.join(root, jpeg_file_name)
                          overlay_file_path = os.path.join(root, overlay_file_name)
                          #print(overlay_file_path)
                          for l in open(overlay_file_path, 'r'):
                              #print(l)
                              larray = l.strip().split(' ')
                              if len(larray) == 2 and larray[0] == 'ABNORMALITY':
                                  total_abnormailties = larray[1]
                              if len(larray) == 6 and larray[0] == 'LESION_TYPE':
                                  lesion_type = larray[1];
                              if len(larray) == 2 and larray[0] == 'PATHOLOGY':
                                  pathology = larray[1];
                          #print pathology , lesion_type
                          patch_file_name = path_patch_dataset + lesion_type + "_" + pathology+"/"
                          #print root , file_name
                          writer.writerow({'root': root, 'file' : file_name, 'lesion_type': lesion_type, 'PATHOLOGY': pathology})
                          extrct_ROIImage(pgm_file_path, jpeg_file_path , patch_file_name, jpeg_file_name)

                      #print 'ending  file', index , patch_file_name , file_name

            except Exception:
                print file_name , root
                pass
