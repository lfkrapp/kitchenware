import numpy as np
import torch as pt


def load_sparse_mask(hgrp, k):
    # get shape
    shape = tuple(hgrp.attrs[k + "_shape"])

    # create map
    M = pt.zeros(shape, dtype=pt.float)
    ids = pt.from_numpy(np.array(hgrp[k]).astype(np.int64))
    M.scatter_(1, ids[:, 1:], 1.0)

    return M


def save_data(hgrp, attrs={}, **data):
    # store data
    for key in data:
        hgrp.create_dataset(key, data=data[key], compression="lzf")

    # save attributes
    for key in attrs:
        hgrp.attrs[key] = attrs[key]


def load_data(hgrp, keys=None):
    # define keys
    if keys is None:
        keys = hgrp.keys()

    # load data
    data = {}
    for key in keys:
        # read data
        data[key] = np.array(hgrp[key])

    # load attributes
    attrs = {}
    for key in hgrp.attrs:
        attrs[key] = hgrp.attrs[key]

    return data, attrs
