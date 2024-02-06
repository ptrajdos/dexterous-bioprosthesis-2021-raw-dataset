import numpy as np

def select_with_classes(X, y, y_sel):
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
