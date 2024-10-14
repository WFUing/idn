"""
logger class for debugging
"""


# todo: fix the bugs of the init process
class Logger:
    """
    This is a logger for reporting debug info
    We will use http api to report debug info in serverless env
    """

    # log level
    DEBUG = 0
    INFO = 1
    LOG_LEVELS = [DEBUG, INFO]

    # log types
    NATIVE = 0
    HTTP = 1
    FILE = 2
    LOGGER_TYPES = [NATIVE]

    log = None  # using logging module by default

    def __init__(self) -> None:
        raise Exception("This is a static class and should not have any instance.")

    @staticmethod
    def use_logger(type, log_level, log_mark="idn"):
        assert type in Logger.LOGGER_TYPES
        if type == Logger.NATIVE:
            from idn.logger.native_logger import NativeLogger
            Logger.log = NativeLogger(log_level, log_mark)

        else:
            raise Exception("Logger type not supported.")

    @staticmethod
    def debug(s):
        Logger.check_logger()
        Logger.log.debug(str(s))

    @staticmethod
    def info(s):
        Logger.check_logger()
        Logger.log.info(str(s))

    @staticmethod
    def finalize():
        Logger.check_logger()
        Logger.log.finalize()

    @staticmethod
    def check_logger():
        if Logger.log is None:
            raise Exception("Logger not init yet, call use_logger() first.")
