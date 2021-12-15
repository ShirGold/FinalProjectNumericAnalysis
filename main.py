import os
import subprocess

from shutil import copyfile
import pandas as pd

def main():
    dirs = set()
    c_dir = os.getcwd()
    c_dir=os.path.join(c_dir,'data_full')
    dir_files=os.listdir(c_dir)
    graders=[]
    for f in dir_files:
        full_path = os.path.join(c_dir, f)
        if os.path.isdir(full_path):
            dirs.add(full_path)
    counter=0
    timeouts=[]
    for dir in dirs:
        copyfile(os.path.join(os.getcwd(),'grader.py'),os.path.join(dir,'grader.py'))
        copyfile(os.path.join(os.getcwd(), 'commons.py'), os.path.join(dir, 'commons.py'))
        copyfile(os.path.join(os.getcwd(), 'functionUtils.py'), os.path.join(dir, 'functionUtils.py'))
        copyfile(os.path.join(os.getcwd(), 'sampleFunctions.py'), os.path.join(dir, 'sampleFunctions.py'))
        print(f'in dir {dir} finished {counter} out of {len(dirs)}')
        if os.path.isfile(os.path.join(dir,'res.csv')):
            os.remove(os.path.join(dir,'res.csv'))
            print('res.removed')			
			#continue
        try:
            sp=subprocess.run(f'python "{os.path.join(dir,"grader.py")}"',timeout=1000)
        except subprocess.TimeoutExpired:
            timeouts.append(dir)
            print(f'dir {dir} has timeout')
        counter+=1
    print('started writing combined csv')
    combined=pd.concat([pd.read_csv(f'{dir}\\res.csv', encoding='utf-16',delimiter='\t') for dir in dirs if os.path.isfile(os.path.join(dir,'res.csv'))])
    combined.to_csv('combined.csv',index=False, encoding='utf-16',sep='\t')
    print('finished writing combined csv')


if __name__ == '__main__':
    main()
