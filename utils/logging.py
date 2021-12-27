import os
import logging

def setup_logging(log_file='log.txt', dummy=False):
    if dummy:
        logging.getLogger('dummy')
        return

    file_mode = 'a' if os.path.isfile(log_file) else 'w'

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    fileout = logging.FileHandler(log_file, mode=file_mode)
    fileout.setLevel(logging.DEBUG)
    fileout.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fileout)