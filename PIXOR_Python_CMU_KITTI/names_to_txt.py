import os

with open("cmu.txt", "w") as a:
    for path, subdirs, files in os.walk(r"KITTI/cmu/image_2"):
       for filename in files:
         f = filename[:-4]
         a.write(str(f)+"\n") 
