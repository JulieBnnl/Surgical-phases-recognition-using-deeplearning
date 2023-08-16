import subprocess
def install_docs():
    package_to_install = "git+https://github.com/tensorflow/docs"

    try:
        subprocess.check_call(["C:/Users/julie/AppData/Local/Microsoft/WindowsApps/python3.11.exe", "-m" ,"pip", "install", package_to_install])
        #subprocess.check_call(["pip", "install", package_to_install])
        print(f"The package {package_to_install} has been successfully installed.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during the installation of the package {package_to_install}.")
        print("Error :", e)