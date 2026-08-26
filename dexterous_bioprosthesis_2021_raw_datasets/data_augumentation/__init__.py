"""Data augmentation package for raw biosignal datasets.

This package provides a collection of signal augmentation techniques for
expanding training datasets of raw biosignals. It includes various
transformations such as noise injection, gain adjustment, time warping,
polarity inversion, pitch shifting, and wavelet-based augmentation.

The augmentation pipeline follows a scikit-learn-compatible interface with
``fit``, ``transform``, ``fit_transform``, and ``sample`` methods.

Subpackages:
    wt_aug: Wavelet transform-based augmentation methods.
"""
