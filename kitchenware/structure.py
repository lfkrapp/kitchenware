import numpy as np

from .dtype import Structure
from .standard_encoding import std_aminoacids, std_backbone, res3to1


def remove_water(structure: Structure) -> Structure:
    # mask for water and deuterium
    m_wat = structure.resnames == "HOH"
    m_hwat = structure.elements == "DOD"

    # remove water
    mask = (~m_wat) & (~m_hwat)

    return structure[mask]


def split_by_chain(structure: Structure) -> dict[str, Structure]:
    # define mask for chains
    cnames = structure.chain_names
    ucnames = np.unique(cnames)
    m_chains = cnames.reshape(-1, 1) == np.unique(cnames).reshape(1, -1)

    # find all interfaces in biounit
    chains: dict[str, Structure] = {}
    for i in range(len(ucnames)):
        # get chain
        chain = structure[m_chains[:, i]]

        # store chain data
        chains[ucnames[i].item()] = chain

    return chains


def concatenate(structures: list[Structure]) -> Structure:
    return Structure(
        **{key: np.concatenate([getattr(structure, key) for structure in structures]) for key in structures[0]}
    )


def concatenate_chains(subunits: dict[str, Structure]) -> Structure:
    # get intersection of keys between chains
    keys = set.intersection(*[set(subunits[cid]) for cid in subunits])

    # concatenate subunits
    structure = Structure(**{key: np.concatenate([getattr(subunits[cid], key) for cid in subunits]) for key in keys})

    return structure


def extract_backbone(structure: Structure) -> Structure:
    # amino-acids and backbone masks
    m_aa = np.isin(structure.resnames, std_aminoacids)
    m_bb = np.isin(structure.names, std_backbone)

    # mask (backbone & polymer residue) or (not polymer residue)
    m = (~m_aa) | (m_aa & m_bb)

    return structure[m]


def split_by_residue(subunit: Structure) -> list[Structure]:
    return [subunit[subunit.resids == i] for i in np.unique(subunit.resids)]


def subunit_to_sequence(subunit: Structure):
    return "".join([res3to1[res.resnames[0]] for res in split_by_residue(subunit) if res.resnames[0] in res3to1])
