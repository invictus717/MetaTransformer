
import os
from os.path import basename

def file_extension(path): 
  return os.path.splitext(path)[1] 
  
def file_name(path): 
  return os.path.splitext(path)[0] 

root = #PATH_TO_DATASET


path = os.listdir(root)  # 6
path.sort()
#vp = 1  # 
file = open(root, 'w')
i = 0
print (path)

for line in path:
    #subdir = root
    #childpath = os.listdir(subdir)
    #mid = int(vp * len(childpath))
    #for child in childpath:
        #subpath = data + '/' + line + '/' + child;
        #d = ' %s' % (i)
    subpath = root+'/'+line
    print (file_extension(subpath))
    
    if file_extension(subpath) == ".pcd":
      print (file_name(line))
      t = file_name(line)
      file.write(t + '\n')
    i = i + 1
    #break
print (i)
file.close()