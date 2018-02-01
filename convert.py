import os
import os
import shutil
import glob
from findROIImage import extrct_ROIImage



path_patch_dataset = "/home/ahtesham/patch_dataset/"
path_to_ddsm = "/home/ahtesham/DDSM/cases/"

template_dir = "../aatif/code/template/mktemplate"
for root, subFolders, file_names in os.walk(path_to_ddsm):
    for file_name in file_names:
#         print "";
        # if ".ics" in file_name:
        #     print root
            # dirname = os.path.basename(root)
            # ics = os.path.join(root, file_name)
            # #shutil.copyfile('../aatif/code/template/case_make_template' , root+'/case_make_template')
            # #shutil.copyfile('../aatif/code/template/mktemplate', root + '/mktemplate')
            # cmd =  '~/aatif/code/template/aq/'+ dirname +'/case_make_template {0}  {1}'.format('~/aatif/code/template/aq/'+dirname+'/'+file_name , dirname)
            # os.system(cmd)
            # os.remove("ChangedFile.csv")
        if ".LJPEG" in file_name:
            ljpeg_path = os.path.join(root, file_name)
            out_path = os.path.join(root, file_name)
            out_path = out_path.split('.LJPEG')[0] + ".jpg"
            print (out_path)
            cmd = './ljpeg.py "{0}" "{1}" --visual --scale 1.0'.format(ljpeg_path, out_path)
            os.system(cmd)

print('done generating pgm files')


# for moving pgm files
# path_to_current_dir = "/home/ahtesham/ljpeg/"
# for root, subFolders, file_names in os.walk(path_to_current_dir):
#     for file_name in file_names:

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
        # if ".LJPEG.1.template.pgm" in file_name:
        #     in_pgm_file_path = os.path.join(root, file_name)
        #     fileInfo = file_name.split('.')
        #     case_dir = fileInfo[0].split('_')[1];
        #     out_pgm_file_path = os.path.join(path_to_ddsm + "case"+case_dir , file_name)
        #     print out_pgm_file_path;
        #     shutil.copyfile(in_pgm_file_path ,out_pgm_file_path);
        #     os.remove(in_pgm_file_path)
print "done moving pgm files"



# for root, subFolders, file_names in os.walk(path_to_ddsm):
#     for file_name in file_names:
#              if ".ics" in file_name:
#                  ics = os.path.join(root, file_name)
#              lesion_type = 'NONE'
#              pathology = 'NONE'
#              total_abnormailties = 'NONE'
#
#              if ".OVERLAY" in file_name:
#                  overlay = os.path.join(root, file_name)
#                  for l in open(overlay, 'r'):
#                      larray = l.strip().split(' ')
#                      if len(larray) == 2 and larray[0] == 'ABNORMALITY' :
#                          total_abnormailties = larray[1]
#                      if len(larray) == 6 and larray[0] == 'LESION_TYPE':
#                          lesion_type = larray[1];
#                      if len(larray) == 2 and larray[0] == 'PATHOLOGY':
#                          pathology = larray[1];
#                  import pdb;
#                  # print '\n'
#              if '.LJPEG.1.template.pgm' in file_name:
#                   fileInfo = file_name.split('.')
#                   jpeg_file_name = fileInfo[0]+'.'+fileInfo[1]+'.jpg';
#                   overlay_file_name = fileInfo[0] + '.' + fileInfo[1] + '.OVERLAY';
#                   pgm_file_path = os.path.join(root, file_name)
#                   jpeg_file_path = os.path.join(root, jpeg_file_name)
#                   overlay_file_path = os.path.join(root, overlay_file_name)
#                   for l in open(overlay_file_path, 'r'):
#                       larray = l.strip().split(' ')
#                       if len(larray) == 2 and larray[0] == 'ABNORMALITY':
#                           total_abnormailties = larray[1]
#                       if len(larray) == 6 and larray[0] == 'LESION_TYPE':
#                           lesion_type = larray[1];
#                       if len(larray) == 2 and larray[0] == 'PATHOLOGY':
#                           pathology = larray[1];
#                   print pathology , lesion_type
#                   patch_file_name = path_patch_dataset + lesion_type + "_" + pathology+"/"
#                   print patch_file_name
#                   extrct_ROIImage(pgm_file_path, jpeg_file_path , patch_file_name, jpeg_file_name)
#
#                   print 'ending first file', file_name






