# loggin and monitoring
import logging
import logging.config
import yaml


import run_test


if __name__ == "__main__" :
    open('./config_logger.yml', 'w').close()
    with open('./config_logger.yml', 'r') as f:
        log_cfg = yaml.safe_load(f.read())
        logging.config.dictConfig(log_cfg)
    logging.info("program starts")
    logging.debug("debug info only")

    run_test.main()

    logging.info("program ends")