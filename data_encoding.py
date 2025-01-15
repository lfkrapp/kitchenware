import numpy as np
import torch as pt

from .dev import Structure
from .standard_encoding import std_elements, std_names, std_resnames
from .structure import chain_name_to_index
from .geometry import secondary_structure


# prepare back mapping
elements_enum = np.concatenate([std_elements, [b"X"]])
names_enum = np.concatenate([std_names, [b"UNK"]])
resnames_enum = np.concatenate([std_resnames, [b"UNX"]])

# prepare config summary
config_encoding = {"std_elements": std_elements, "std_resnames": std_resnames, "std_names": std_names}


def onehot(x, v):
    m = x.reshape(-1, 1) == np.array(v).reshape(1, -1)
    return np.concatenate([m, ~np.any(m, axis=1).reshape(-1, 1)], axis=1)


def encode_structure(structure: Structure, device=pt.device("cpu")):
    # coordinates
    X = pt.from_numpy(structure.xyz.astype(np.float32)).to(device)

    # atom to residues mapping
    resids = pt.from_numpy(structure.resids).to(device)
    Mr = (resids.unsqueeze(1) == pt.unique(resids).unsqueeze(0)).float()

    # atom to chain mapping
    cids = pt.from_numpy(chain_name_to_index(structure)).to(device)
    Mc = (cids.unsqueeze(1) == pt.unique(cids).unsqueeze(0)).float()

    # charge features
    qe = pt.from_numpy(onehot(structure.elements, std_elements).astype(np.float32)).to(device)
    qr = pt.from_numpy(onehot(structure.resnames, std_resnames).astype(np.float32)).to(device)
    qn = pt.from_numpy(onehot(structure.names, std_names).astype(np.float32)).to(device)

    return X, qe, qr, qn, Mr, Mc


def extract_secondary_structure_map(X, qe, qn, Mr):
    # assign secondary structures
    m_ca = (qn[:, 0] > 0.5) & (qe[:, 0] > 0.5)
    ss = secondary_structure(X[m_ca])

    # indices of secondary structure segments
    ids_ss_seg = pt.cat([pt.zeros(1, device=ss.device, dtype=ss.dtype), pt.cumsum(pt.abs(pt.diff(ss)), dim=0)])

    # construct mapping matrix at residue level and reshape at atom level
    Ms = pt.zeros((Mr.shape[1], int(ids_ss_seg[-1].item()) + 1), dtype=pt.float32)
    Ms[pt.where(Mr[m_ca] > 0.5)[0], ids_ss_seg] = 1.0
    Ms = pt.matmul(Mr, Ms)

    return Ms


def extract_topology(X, num_nn):
    # compute displacement vectors
    R = X.unsqueeze(0) - X.unsqueeze(1)
    # compute distance matrix
    D = pt.norm(R, dim=2)
    # mask distances
    D = D + 2.0 * pt.max(D) * (D < 1e-2).float()
    # normalize displacement vectors
    R = R / D.unsqueeze(2)

    # find nearest neighbors
    knn = min(num_nn, D.shape[0])
    D_topk, ids_topk = pt.topk(D, knn, dim=1, largest=False)
    R_topk = pt.gather(R, 1, ids_topk.unsqueeze(2).repeat((1, 1, X.shape[1])))

    return ids_topk, D_topk, R_topk, D, R
