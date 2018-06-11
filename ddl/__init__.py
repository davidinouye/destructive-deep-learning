# Nothing needed in here for now
import logging


try:
    from logging import NullHandler
except ImportError:
    from logging import Handler

    # Simple class for backwards compatibility with Python 2
    class NullHandler(Handler):
        def emit(self, record):
            pass


logging.getLogger(__name__).addHandler(NullHandler())
