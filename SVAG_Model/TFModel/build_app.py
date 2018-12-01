import os
import subprocess

# ANDROID


def build_android_app(model_file):
    if not os.path.exists(model_file):
        return False, "model file not exists"
    return True, "Success"