import io
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


def serialize_tensor(x: pt.Tensor) -> bytes:
    # convert to bytes
    buffer = io.BytesIO()
    pt.save(x.cpu(), buffer)

    return buffer.getvalue()


def deserialize_tensor(buffer: bytes) -> pt.Tensor:
    return pt.load(io.BytesIO(buffer), weights_only=True)


def encode_sparse_mask(M: pt.Tensor) -> pt.Tensor:
    return pt.cat(
        [
            pt.tensor([M.shape]),
            pt.stack(pt.where(M.cpu().bool()), dim=1),
        ],
        dim=0,
    ).short()


def decode_sparse_mask(mids):
    M = pt.zeros(mids[0, 0], mids[0, 1], dtype=pt.float)
    M[mids[1:, 0].long(), mids[1:, 1].long()] = 1.0
    return M
