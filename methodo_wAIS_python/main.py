# loggin and monitoring
import logging
import logging.config
import yaml
from pprint import pprint
from utils.log import logstr
from logging import info, debug, warn, error


import run_test

import os


if __name__ == "__main__" :
    os.system("clear")
    
    open('log/debug.log', 'w').close()
    open('log/info.log', 'w').close()
    
    with open('./config_logger.yml', 'r') as f:
        log_cfg = yaml.safe_load(f.read())
        logging.config.dictConfig(log_cfg)
    info(logstr("program starts"))
    debug(logstr("debug info only"))

    run_test.main()

    info("program ends")