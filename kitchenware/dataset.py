import numpy as np
import torch as pt

from .dev import Structure
from .structure_io import read_structure as read_pdb
from .structure import clean_structure, tag_hetatm_chains


def select_by_sid(dataset, sids_sel):
    # extract sids of dataset
    sids = np.array(["_".join([s.split(":")[0] for s in key.split("/")[1::2]]) for key in dataset.keys])

    # create selection mask
    m = np.isin(sids, sids_sel)

    return m


def select_by_max_ba(dataset, max_ba):
    # extract aids of dataset
    aids = np.array([int(key.split("/")[2]) for key in dataset.keys])

    # create selection mask
    m = aids <= max_ba

    return m


def select_complete_assemblies(dataset, m):
    # get non-selected subunits
    rmkeys = np.unique(dataset.keys[~m])

    # select all assemblies not containing non-selected subunits
    return ~np.isin(dataset.rkeys, rmkeys)


def select_by_interface_types(dataset, l_types, r_types):
    # get types id
    t0 = np.where(np.isin(dataset.mids, l_types))[0]
    t1 = np.where(np.isin(dataset.mids, r_types))[0]

    # ctypes selection mask
    cm = np.isin(dataset.ctypes[:, 1], t0) & np.isin(dataset.ctypes[:, 2], t1)

    # apply selection on dataset
    m = np.isin(np.arange(dataset.keys.shape[0]), dataset.ctypes[cm, 0])

    return m


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


class StructuresDataset(pt.utils.data.Dataset):
    def __init__(self, pdb_filepaths, with_preprocessing=True):
        super(StructuresDataset).__init__()
        # store dataset filepath
        self.pdb_filepaths = pdb_filepaths

        # store flag
        self.with_preprocessing = with_preprocessing

    def __len__(self):
        return len(self.pdb_filepaths)

    def __getitem__(self, i) -> tuple[Structure | None, str]:
        # find pdb filepath
        pdb_filepath = self.pdb_filepaths[i]

        # parse pdb
        try:
            structure = read_pdb(pdb_filepath)
        except Exception as e:
            print(f"ReadError: {pdb_filepath}: {e}")
            return None, pdb_filepath

        if self.with_preprocessing:
            # process structure
            structure = clean_structure(structure)

            # update molecules chains
            structure = tag_hetatm_chains(structure)

            return structure, pdb_filepath
        else:
            return structure, pdb_filepath
