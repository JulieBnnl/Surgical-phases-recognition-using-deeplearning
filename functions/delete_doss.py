import os
import shutil

def delete_doss(path):
    print(f"Deleting {path} folder...")
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                shutil.rmtree(dir_path)
    print(f"{path} successfully delete !")