import os
import subprocess
from shutil import copyfile

ANDROID = '../../ANDROID/gradlew'
ASSETS = '../../ANDROID/assets'
OUTPUT = '../../ANDROID/gradleBuild/output/apk/debug'
APP_FILE = 'ANDROID-debug.apk'

def build_android_app(labels, saved_models_path, model_file):
    if not os.path.exists(model_file):
        return False, "model file not exists"
    with open(os.path.join(saved_models_path, 'labels_info.txt'), 'w') as file:
        s = '\n'.join(labels)
        file.write(s)
    copyfile(os.path.join(saved_models_path, 'labels_info.txt'), os.path.join(ASSETS, 'labels_info.txt'))
    copyfile(model_file, os.path.join(ASSETS, 'test.pb'))
    copyfile(os.path.join(OUTPUT, APP_FILE), os.path.join(saved_models_path, APP_FILE))
    return True, os.path.join(saved_models_path, APP_FILE)