import shutil
import os, os.path


def require_empty(d, recreate=False):
    if os.path.exists(d):
        if recreate:
            shutil.rmtree(d)
        else:
            raise OSError('Path {} exists and no --overwrite set. Exiting'.format(d))
    os.makedirs(d)