import numpy as np
import torch as pt
import numpy.typing as npt

from .dev import Structure
from .standard_encoding import std_aminoacids, std_backbone, res3to1
from .structure_io import read_structure as read_pdb
from .geometry import locate_contacts


def clean_structure(structure: Structure, rm_hetatm=False, rm_wat=True):
    # mask for water, hydrogens and deuterium
    m_wat = structure.resnames == "HOH"
    m_h = structure.elements == "H"
    m_d = structure.elements == "D"
    m_hwat = structure.elements == "DOD"

    if rm_wat:
        # remove water
        mask = (~m_wat) & (~m_h) & (~m_d) & (~m_hwat)
    else:
        # keep but tag water
        mask = (~m_h) & (~m_d) & (~m_hwat)
        structure.resids[m_wat] = -999

    if rm_hetatm:
        # remove hetatm
        mask &= structure.het_flags == "A"

    # filter structure atoms
    for key in structure:
        structure[key] = structure[key][mask]

    # find changes due to chain
    chains = structure.chain_names
    ids_chains = np.where(np.array(chains).reshape(-1, 1) == np.unique(chains).reshape(1, -1))[1]
    delta_chains = np.abs(np.sign(np.concatenate([[0], np.diff(ids_chains)])))

    # find changes due to inertion code
    icodes = structure.icodes
    ids_icodes = np.where(np.array(icodes).reshape(-1, 1) == np.unique(icodes).reshape(1, -1))[1]
    delta_icodes = np.abs(np.sign(np.concatenate([[0], np.diff(ids_icodes)])))

    # find changes due to resids
    resids = structure.resids
    delta_resids = np.abs(np.sign(np.concatenate([[0], np.diff(resids)])))

    # renumber resids
    resids = np.cumsum(np.sign(delta_chains + delta_resids + delta_icodes)) + 1

    # update resids
    structure.resids = resids

    # return process structure
    return structure


def atom_select(structure: Structure, sel) -> Structure:
    return Structure(**{key: structure[key][sel] for key in structure})


def split_by_chain(structure: Structure) -> dict[str, Structure]:
    # define mask for chains
    cnames = structure.chain_names
    ucnames = np.unique(cnames)
    m_chains = cnames.reshape(-1, 1) == np.unique(cnames).reshape(1, -1)

    # find all interfaces in biounit
    chains: dict[str, Structure] = {}
    for i in range(len(ucnames)):
        # get chain
        chain = atom_select(structure, m_chains[:, i])

        # store chain data
        chains[ucnames[i]] = chain

    return chains


def extract_backbone(structure: Structure) -> Structure:
    # amino-acids and backbone masks
    m_aa = np.isin(structure.resnames, std_aminoacids)
    m_bb = np.isin(structure.names, std_backbone)

    # mask (backbone & polymer residue) or (not polymer residue)
    m = (~m_aa) | (m_aa & m_bb)

    return atom_select(structure, m)


def split_by_residue(subunit: Structure) -> list[Structure]:
    return [atom_select(subunit, subunit.resids == i) for i in np.unique(subunit.resids)]


def subunit_to_sequence(subunit: Structure):
    return "".join([res3to1[res.resnames[0]] for res in split_by_residue(subunit) if res.resnames[0] in res3to1])


def concatenate_chains(subunits: dict[str, Structure]) -> Structure:
    # get intersection of keys between chains
    keys = set.intersection(*[set(subunits[cid]) for cid in subunits])

    # concatenate subunits
    structure = Structure(**{key: np.concatenate([subunits[cid][key] for cid in subunits]) for key in keys})

    return structure


def tag_hetatm_chains(structure: Structure):
    # get hetatm
    m_hetatm = structure.het_flags == "H"
    resids_hetatm = structure.resids[m_hetatm]

    # split if multiple hetatm
    delta_hetatm = np.cumsum(np.abs(np.sign(np.concatenate([[0], np.diff(resids_hetatm)]))))

    # update chain name
    cids_hetatm = np.array([f"{cid}:{hid}" for cid, hid in zip(structure.chain_names[m_hetatm], delta_hetatm)])
    cids = structure.chain_names.copy().astype(np.dtype("<U10"))
    cids[m_hetatm] = cids_hetatm
    structure.chain_names = np.array(list(cids)).astype(str)

    return structure


def chain_name_to_index(structure: Structure) -> npt.NDArray[int]:
    # get chain names
    cnames = structure.chain_names

    # convert it to index
    cids = np.where(cnames.reshape(-1, 1) == np.unique(cnames).reshape(1, -1))[1]

    return cids


def extract_all_contacts(subunits, r_thr, device=pt.device("cpu")):
    # get subunits names
    snames = list(subunits)

    # extract interfaces
    contacts_dict = {}
    for i in range(len(snames)):
        # current selection chain
        cid_i = snames[i]

        for j in range(i + 1, len(snames)):
            # current selection chain
            cid_j = snames[j]

            # find contacts
            ids_i, ids_j, d_ij = (
                out.cpu()
                for out in locate_contacts(
                    pt.from_numpy(subunits[cid_i].xyz).to(device),
                    pt.from_numpy(subunits[cid_j].xyz).to(device),
                    r_thr,
                )
            )

            # insert contacts
            if (ids_i.shape[0] > 0) and (ids_j.shape[0] > 0):
                if f"{cid_i}" in contacts_dict:
                    contacts_dict[f"{cid_i}"].update({f"{cid_j}": {"ids": pt.stack([ids_i, ids_j], dim=1), "d": d_ij}})
                else:
                    contacts_dict[f"{cid_i}"] = {f"{cid_j}": {"ids": pt.stack([ids_i, ids_j], dim=1), "d": d_ij}}

                if f"{cid_j}" in contacts_dict:
                    contacts_dict[f"{cid_j}"].update({f"{cid_i}": {"ids": pt.stack([ids_j, ids_i], dim=1), "d": d_ij}})
                else:
                    contacts_dict[f"{cid_j}"] = {f"{cid_i}": {"ids": pt.stack([ids_j, ids_i], dim=1), "d": d_ij}}

    return contacts_dict


def data_to_structure(X, q, Mr, Mc, std_elements, std_resnames, std_names) -> Structure:
    # q = [qe, qr, qn]
    # resnames
    resnames_enum = np.concatenate([std_resnames, [b"UNX"]])
    q_resnames = q[:, len(std_elements) + 1 : len(std_elements) + len(std_resnames) + 2]
    resnames = resnames_enum[np.where(q_resnames)[1]]

    # names
    q_names = q[:, len(std_elements) + len(std_resnames) + 2 :]
    names_enum = np.concatenate([std_names, [b"UNK"]])
    names = names_enum[np.where(q_names)[1]]

    # elements
    q_elements = q[:, : len(std_elements) + 1]
    elements_enum = np.concatenate([std_elements, [b"X"]])
    elements = elements_enum[np.where(q_elements)[1]]

    # resids
    ids0, ids1 = np.where(Mr > 0.5)
    resids = np.zeros(Mr.shape[0], dtype=np.int32)
    resids[ids0] = ids1 + 1

    # chains
    ids0, ids1 = np.where(Mc > 0.5)
    cids = np.zeros(Mc.shape[0], dtype=np.int64)
    cids[ids0] = ids1 + 1

    # hetatm flag
    het_flags = np.array(["H" if rn == "UNX" else "A" for rn in resnames])

    # pack subunit struct
    return Structure(
        xyz=X,
        names=names,
        elements=elements,
        resnames=resnames,
        resids=resids,
        chain_names=cids.astype(str),
        het_flags=het_flags,
        icodes=np.array([""] * X.shape[0]),
        bfactors=np.zeros(X.shape[0], dtype=np.float32),
    )


def encode_bfactor(structure: Structure, p):
    # C_alpha mask
    names = structure.names
    elements = structure.elements
    m_ca = (names == "CA") & (elements == "C") & (np.isin(structure.resnames, std_aminoacids))
    resids = structure.resids

    if p.shape[0] == m_ca.shape[0]:
        structure.bfactors = p

    elif p.shape[0] == np.sum(m_ca):
        # expand c_alpha bfactor to all
        bf = np.zeros(len(resids), dtype=np.float32)
        for i in np.unique(resids):
            m_ri = resids == i
            i_rca = np.where(m_ri[m_ca])[0]
            if len(i_rca) > 0:
                bf[m_ri] = float(np.max(p[i_rca]))

        # store result
        structure.bfactors = bf

    elif p.shape[0] == np.unique(resids).shape[0]:
        # expand c_alpha bfactor to all
        uresids = np.unique(resids)
        bf = np.zeros(len(resids), dtype=np.float32)
        for i in uresids:
            m_ri = resids == i
            m_uri = uresids == i
            bf[m_ri] = float(np.max(p[m_uri]))

        # store result
        structure.bfactors = bf

    else:
        print("WARNING: bfactor not saved")

    return structure


def process_structure(structure: Structure, rm_hetatm=False, rm_wat=True) -> Structure:
    # keep original resids
    # structure["resid_orig"] = structure["resid"].copy()

    # process structure
    structure = clean_structure(structure, rm_hetatm=rm_hetatm, rm_wat=rm_wat)

    # update molecules chains
    structure = tag_hetatm_chains(structure)

    # change chain name to chain index
    # cids = chain_name_to_index(structure)

    return structure


def load_structure(pdb_filepath, rm_hetatm=False, rm_wat=True) -> Structure:
    # read structure
    structure = read_pdb(pdb_filepath)

    # process structure
    structure = process_structure(structure, rm_hetatm=rm_hetatm, rm_wat=rm_wat)

    return structure
