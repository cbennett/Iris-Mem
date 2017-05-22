# -*- coding: utf-8 -*-

"""Personal cookbook: a collection of custom helper functions.
"""

from __future__ import (division, print_function, unicode_literals,
                        absolute_import)

import numpy as np
from sklearn import datasets

try:
    from tqdm import tqdm
except:  # not specifying the excepttion as it change in Python 3
    print("Warning! Did not succeeded in importing 'tqdm': " +
          "falling back to a simple pass through function...")
    def tqdm(iterable, **kwargs):
        return iter(iterable)


def load_iris(vmin=-1, vmax=1, *args, **kwargs):
    """Load the Iris dataset.

    Parameters
    ----------
    vmin, vmax : floats or None, optional, default: -1, 1
        The lower, resp. upper, boundaries of the range where the data
        values will be rescaled (i.e. [vmin..vmax]).  If one of these
        boundaries is None, it will be replaced with the most extrem
        corresponding value among **all** the sample.

    *args, **kwargs :
        *Not implemented at the moment.*

    Returns
    -------
    iris : object returned by `sklearn.datasets.load_iris`
        The modified object that one usually uses with scikit-learn
        (should be a drop-in replacement).

    Remarks
    -------
    The Iris dataset has 4 features (``nfts = 4``) and
    3 classes (``ncls = 3``).  The whole dataset is made
    of 150 samples: ``iris.data.shape = (150, 4)``.
    """

    dset = datasets.load_iris()

    if vmin is None:
        vmin = np.min(dset.data)

    if vmax is None:
        vmax = np.max(dset.data)

    fts_mins = np.min(dset.data, axis=0)
    fts_ptps = np.ptp(dset.data, axis=0)  # ptp stands for "peak-to-peak"
    dset.data = (vmax - vmin) * ((dset.data - fts_mins)/fts_ptps - 0.5)

    return dset


def load_digits(vmin=-1, vmax=1, *args, **kwargs):
    """Load the handwritten digits dataset.  **This is not the MNIST
    dataset!**  See `load_mnist` if that is what you are looking for.

    Parameters
    ----------
    vmin, vmax : floats or None, optional, default: -1, 1
        The lower, resp. upper, boundaries of the range where the data
        values will be rescaled (i.e. [vmin..vmax]).  If one of these
        boundaries is None, it will be replaced with the most extrem
        corresponding value among **all** the sample.

    *args, **kwargs :
        *Not implemented at the moment.*

    Returns
    -------
    digits : object returned by `sklearn.datasets.load_digits`
        The modified object that one usually uses with scikit-learn
        (should be a drop-in replacement).

    Remarks
    -------
    The handwritten digits dataset has 64 features (``nfts = 64``)
    and 10 classes (``ncls = 10``).  The whole dataset is made of
    1797 samples: ``digits.data.shape = (1797, 64)``.  Each sample
    correspond to a grayscale 8x8 pixel-frame, taking originally
    (float) values in the **integer** range [0..16].
    """

    dset = datasets.load_digits()

    if vmin is None:
        vmin = np.min(dset.data)

    if vmax is None:
        vmax = np.max(dset.data)

    fts_mins = np.min(dset.data, axis=1)[:, np.newaxis]
    fts_ptps = np.ptp(dset.data, axis=1)[:, np.newaxis]
    dset.data = (vmax - vmin) * ((dset.data - fts_mins)/fts_ptps - 0.5)

    return dset



