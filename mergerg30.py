import shutil
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3: exit(0)
    path = sys.argv[1]
    out_path = sys.argv[2]
    for set_name in [fn for fn in os.listdir(path) if fn.startswith('Set ')]:
        set_path = path + '/' + set_name
        for instance_name in [fn for fn in os.listdir(set_path) if fn.endswith('.rcp')]:
            instance_path = set_path + '/' + instance_name
            shutil.copy(instance_path, out_path+'/'+set_name.replace('Set ','Set')+'_'+instance_name)
