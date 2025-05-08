import os
import os.path as osp
import time
import yaml

from .tools import mkdir_if_missing


class Logger(object):
    """Write output to console and external text file

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
        if fpath is not None:
            # 使用时间戳生成唯一的日志文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            fpath = osp.join(osp.dirname(fpath), f"{osp.basename(fpath).split('.')[0]}_{timestamp}.log") 
        
        self.fpath = fpath
        self.fdir = osp.dirname(fpath) if fpath is not None else None

        self.file = None
        if fpath is not None:
            mkdir_if_missing(self.fdir)
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg: str, timestamp: bool = False, end: str = '\n'):
        """Write message to console and file

        Args:
            msg (str): message to be written
            timestamp (bool): whether to add timestamp
            end (str): end character

        Examples:
            >>> logger.write('Hello, world!')
        """
        msg = str(msg)
        
        if timestamp:
            time_stamp = time.strftime(
                "%Y-%m-%d %H:%M:%S", 
                time.localtime())
            msg = f'[{time_stamp}] {msg}'

        msg += end
        print(msg, end='')
        if self.file is not None:
            self.file.write(msg)
        self.flush()

    def info(self, msg: str, end: str = '\n'):
        """Write message to console and file with timestamp

        Args:
            msg (str): message to be written
            end (str): end character

        Examples:
            >>> logger.info('Hello, world!')
        """
        self.write(msg, timestamp=True, end=end)

    def create_config(self, cfg: dict):
        """Create config file and save to disk in the form of yaml

        Args:
            cfg (dict): arguments to be wrote
        """
        fpath = osp.join(self.fdir, 'cfg.yaml')

        with open(fpath, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, allow_unicode=True)

    def flush(self):
        """Flush the buffer to external file

        Examples:
            >>> logger.flush()
        """
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        """Close the file and console

        Examples:
            >>> logger.close()
        """
        if self.file is not None:
            self.file.close()
