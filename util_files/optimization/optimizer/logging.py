import io
import logging


class Logger(logging.Logger):
    @classmethod
    def prepare_logger(cls, loglevel='debug', logger_id='default_logger', logfile=None):
        loglevel = Logger.get_log_level(loglevel)
        logging.setLoggerClass(cls)
        logger = logging.getLogger(logger_id)
        logger.setLevel(loglevel)

        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(loglevel)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.info_stream = StreamHandler(logger, logging.INFO)

        if logfile is not None:
            file_handler = logging.FileHandler(logfile)
            file_handler.setLevel(loglevel)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def get_log_level(loglevel):
        if isinstance(loglevel, str):
            loglevel = {
                'critical': 50,
                'error': 40,
                'warning': 30,
                'info': 20,
                'debug': 10
            }[loglevel]

        if loglevel >= 50:
            return logging.CRITICAL
        elif loglevel >= 40:
            return logging.ERROR
        elif loglevel >= 30:
            return logging.WARNING
        elif loglevel >= 20:
            return logging.INFO
        elif loglevel >= 10:
            return logging.DEBUG
        else:
            return logging.NOTSET


class StreamHandler(io.StringIO):
    def __init__(self, logger, level):
        super().__init__()
        self.logger = logger
        self.level = level
        self.buf = ''

    def write(self, buf):
        self.buf = buf.strip('\r\n')

    def flush(self):
        self.logger.log(self.level, self.buf)

