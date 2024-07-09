"""
Logger
version: 0.0.1
update: 2023.12.13
"""
import os
import time
import yaml

_logger = None

def get_logger():
    global _logger
    return _logger


class Logger():
    def __init__(self, fdir: str):
        self.fdir = fdir
        self.fpath = os.path.join(self.fdir, 'output.log')

        if not os.path.exists(self.fdir):
            os.makedirs(self.fdir)
        
        global _logger
        _logger = self

    def info(self, content: str):
        """
        Record information and write to log file
        Args:
            content (str):  the content to be recorded
        """
        print(content)

        with open(self.fpath, 'a') as f:
            for c in str(content).split('\n'):
                time_stamp = time.strftime(
                    "%Y-%m-%d %H:%M:%S", 
                    time.localtime())

                c = f'[{time_stamp}] {c}\n'
                f.write(c)

    def create_config(self, cfg: dict):
        """
        Create config file and save to disk in the form of yaml
        Args:
            cfg (dict):     arguments to be wrote
        """
        fpath = os.path.join(self.fdir, 'cfg.yaml')
        
        with open(fpath, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, allow_unicode=True)
