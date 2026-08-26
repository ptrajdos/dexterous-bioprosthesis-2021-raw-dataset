"""Module providing numpy dataset selection utilities.

Provides :func:`select_with_classes` for filtering by class labels.
"""
import numpy as np

def select_with_classes(X, y, y_sel):
    """Select dataset rows matching the specified class labels."""
    XS = None
    ys = []
    first = True
    for y_label in y_sel:
        if first:
            XS = X[y == y_label, :]
            first = False
        else:
            XS = np.vstack((XS, X[y == y_label, :]))

        sel = list(y[y == y_label])
        ys += sel

    ys = np.asanyarray(ys)

    return XS, ys
