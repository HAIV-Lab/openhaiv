import os
import os.path as osp
import errno


def mkdir_if_missing(dirname):
    """Create dirname if it is missing.

    Args:
        dirname (str): directory path
    """
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
