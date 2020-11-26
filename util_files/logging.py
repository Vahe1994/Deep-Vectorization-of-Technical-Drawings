import time
from contextlib import contextmanager
import logging


class Logger(logging.Logger):
    @contextmanager
    def print_duration(self, duration_of_what, print_start=False):
        if print_start:
            self.debug(duration_of_what)
        start = time.clock()
        yield
        duration = time.clock() - start
        self.debug(duration_of_what + ' complete in {0:.3f} seconds'.format(duration))

    def info_scalars(self, s, scalars_dict, **kwargs):
        for key, value in scalars_dict.items():
            log_s = s.format(key=key, value=value, **kwargs)
            self.info(log_s)

    def info_trainable_params(self, torch_model):
        self.info('Total number of trainable parameters for {model}: {params}'.format(
            model=torch_model.__class__,
            params=sum(p.numel() for p in torch_model.parameters() if p.requires_grad)))


def create_logger(options):
    loglevel = logging.DEBUG if options.verbose else logging.INFO
    logging.setLoggerClass(Logger)
    logger = logging.getLogger('train')
    logger.setLevel(loglevel)

    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s: %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(loglevel)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if options.logging_filename:
        file_handler = logging.FileHandler(options.logging_filename)
        file_handler.setLevel(loglevel)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
