import sys
import time
import os
import os.path as osp

from .tools import mkdir_if_missing


class Logger(object):
    """Write console output to external text file.

    Args:
        fpath (str): directory to save logging file.
            If None, do not write to external file.

    Examples:
        >>> import sys
        >>> import os.path as osp
        >>> save_dir = 'output/experiment-1'
        >>> log_name = 'train.log'
        >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """
    def __init__(self, fpath: str | None = None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg: str, timestamp: bool = True, end: str = '\n'):
        """Write message to console and file

        Args:
            msg (str): message to be written
            timestamp (bool): whether to add timestamp
            end (str): end character

        Examples:
            >>> logger.write('Hello, world!')
        """
        if timestamp:
            time_stamp = time.strftime(
                "%Y-%m-%d %H:%M:%S", 
                time.localtime())
            msg = f'[{time_stamp}] {msg}'

        msg += end
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        """Flush the buffer to external file

        Examples:
            >>> logger.flush()
        """
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        """Close the file and console

        Examples:
            >>> logger.close()
        """
        self.console.close()
        if self.file is not None:
            self.file.close()
