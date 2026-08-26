"""Module providing numba compatibility utilities.

Defines a conditional ``jit`` decorator that falls back to a no-op
when numba is not available.
"""
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        """Conditional numba JIT decorator; no-op if numba is unavailable."""
        def decorator(func):
            """Return the function unchanged (no-op decorator)."""
            return func
        return decorator
