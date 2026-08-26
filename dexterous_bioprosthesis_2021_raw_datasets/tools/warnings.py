"""Module providing custom warning utilities.

Defines :func:`custom_formatwarning` for formatting warnings with
traceback information.
"""
import traceback
import warnings
import sys


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    """Format a warning message with traceback information."""
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))