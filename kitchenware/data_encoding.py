import numpy as np
import torch as pt

from .dtype import Structure, StructureData
from .standard_encoding import std_elements, std_names, std_resnames
from .metrics import secondary_structure


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
    cids = pt.from_numpy(
        np.where(structure.chain_names.reshape(-1, 1) == np.unique(structure.chain_names).reshape(1, -1))[1]
    ).to(device)
    Mc = (cids.unsqueeze(1) == pt.unique(cids).unsqueeze(0)).float()

    # charge features
    qe = pt.from_numpy(onehot(structure.elements, std_elements).astype(np.float32)).to(device)
    qr = pt.from_numpy(onehot(structure.resnames, std_resnames).astype(np.float32)).to(device)
    qn = pt.from_numpy(onehot(structure.names, std_names).astype(np.float32)).to(device)

    return StructureData(X=X, qe=qe, qr=qr, qn=qn, Mr=Mr, Mc=Mc)


def encode_secondary_structure(data: StructureData) -> tuple[pt.Tensor, pt.Tensor]:
    # compute secondary structure from carbon-alpha coordinates
    m_ca = (data.qn[:, 0] > 0.5) & pt.any(data.qr[:, :20] > 0.5, dim=1)
    ca_xyz = data.X[m_ca]
    if ca_xyz.shape[0] < 5:
        qs = pt.zeros((data.Mr.shape[1], 4), device=m_ca.device)
        qs[:, 3] = 1.0
    else:
        # compute secondary structure
        ss = secondary_structure(data.X[m_ca])

        # encode secondary structure feature
        qs = pt.zeros((data.Mr.shape[1], 4), device=ss.device)
        m_aa = pt.sum(data.Mr[m_ca], dim=0) > 0.5
        qs[m_aa, ss] = 1.0
        qs[~m_aa, 3] = 1.0

    # mapping between atoms and seconary structure segement
    sids = pt.cumsum(
        pt.cat([pt.tensor([0], dtype=pt.long, device=qs.device), pt.max(pt.diff(qs.long(), dim=0), dim=1)[0]]), 0
    )
    Ms = (sids.unsqueeze(1) == pt.unique(sids).unsqueeze(0)).float()

    # broadcast to atom level
    qs = pt.matmul(data.Mr, qs)
    Ms = pt.matmul(data.Mr, Ms)

    return qs, Ms


def remove_clash(data: StructureData, d_thr=0.75):
    # compute distance matrix and mask upper to keep first instance of atom with clash
    D = pt.norm(data.X.unsqueeze(0) - data.X.unsqueeze(1), dim=2)
    D = D + pt.triu(pt.ones_like(D)) * pt.max(D) * 2.0

    # detect atom clashing using distance threshold
    m_clash = pt.any(D < d_thr, dim=1)

    # extend to whole molecule
    m_clash_ext = pt.sum(data.Mr[:, pt.sum(data.Mr[m_clash], dim=0) > 0.5], dim=1) > 0.5

    return data[~m_clash_ext]


def data_to_structure(data: StructureData) -> Structure:
    # elements
    elements_enum = np.concatenate([std_elements, [b"X"]])
    elements = elements_enum[np.where(data.qe.cpu().numpy())[1]]

    # names
    names_enum = np.concatenate([std_names, [b"UNK"]])
    names = names_enum[np.where(data.qn.cpu().numpy())[1]]

    # resnames
    resnames_enum = np.concatenate([std_resnames, [b"UNX"]])
    resnames = resnames_enum[np.where(data.qr.cpu().numpy())[1]]

    # resids
    ids0, ids1 = np.where(data.Mr.cpu().numpy() > 0.5)
    resids = np.zeros(data.Mr.shape[0], dtype=np.int32)
    resids[ids0] = ids1 + 1

    # chains
    ids0, ids1 = np.where(data.Mc.cpu().numpy() > 0.5)
    cids = np.zeros(data.Mc.shape[0], dtype=np.int64)
    cids[ids0] = ids1 + 1

    # pack subunit struct
    return Structure(
        xyz=data.X.cpu().numpy(),
        names=names,
        elements=elements,
        resnames=resnames,
        resids=resids,
        chain_names=cids.astype(str),
    )
