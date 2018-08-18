"""Init file for ddl package."""
# Nothing needed in here for now
import logging

try:
    from logging import NullHandler
except ImportError:
    from logging import Handler

    # Simple class for backwards compatibility with Python 2
    class NullHandler(Handler):
        """Null logging handler."""

        def emit(self, record):
            """Do not emit anything for this null handler.

            Parameters
            ----------
            record :

            """
            pass


logging.getLogger(__name__).addHandler(NullHandler())
