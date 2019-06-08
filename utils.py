import errno
import os
import zipfile


def mkdirp(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def extract_saved(tmp_folder, filepath):
    has_saved_checkpoint = False
    with zipfile.ZipFile(filepath, 'r') as zip_file:
        namelist = zip_file.namelist()
        for name in namelist:
            if name == 'checkpoint':
                has_saved_checkpoint = True
            filecontent = zip_file.read(name)
            filepath = os.path.join(tmp_folder, name)
            with open(filepath, 'wb') as f:
                f.write(filecontent)
    return has_saved_checkpoint
