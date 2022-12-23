import logging

def logstr(obj):
    return f"\n{obj}\n"

def info(obj) -> None:
    logging.info(logstr(obj))
    
def debug(obj) -> None:
    logging.debug(logstr(obj))
    
def warn(obj) -> None:
    logging.warn(logstr(obj))
    
def critical(obj) -> None:
    logging.critical(logstr(obj))

def error(obj) -> None:
    logging.error(logstr(obj))