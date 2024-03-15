import logging, time, os

def get_log_path(args):
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    
    current_time=time.strftime('%Y%m%d%H%M',time.localtime(time.time() )) 
    
    log_txt_name = args.model_name + "_" + args.sc_vul_type + "_" + current_time +".txt"
    log_txt_path = os.path.join(args.log_dir, log_txt_name)
    return log_txt_path, log_txt_name


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
