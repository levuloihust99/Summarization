class UnsupportedModeError(Exception):
    """Raised when an unsupported mode is provided."""

class BufferExceedError(Exception):
    """Raised when one single record has larger size than buffer size."""

class NotEnoughMaxLinesPerFile(Exception):
    """Raised when max lines per output file is too small, i.e. smaller than 3."""
