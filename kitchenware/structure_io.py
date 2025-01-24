import gemmi
import numpy as np
import torch as pt

from .dtype import Structure
from .structure import split_by_chain
from .standard_encoding import std_resnames


def load_structure(filepath: str, rm_wat=False, rm_hs=True) -> Structure:
    # use gemmi to parse file
    doc = gemmi.read_structure(filepath)

    # alternative location and insertion sets
    altloc_set, resid_set = set(), set()

    # data storage
    xyz_l, element_l, name_l, resname_l, resid_l, chain_name_l = [], [], [], [], [], []

    # parse structure
    for _, model in enumerate(doc):
        for a in model.all():
            # skip hydrogens and deuterium
            if rm_hs and ((a.atom.element.name == "H") or (a.atom.element.name == "D")):
                continue

            # skip (heavy) water
            if rm_wat and ((a.residue.name == "HOH") or (a.residue.name == "DOD")):
                continue

            # altloc check (keep first encountered)
            if a.atom.has_altloc():
                key = f"{a.chain.name}_{a.residue.seqid.num}_{a.residue.name}_{a.atom.name}"
                if key in altloc_set:
                    continue
                else:
                    altloc_set.add(key)

            # insertion code (shift residue index)
            resid_set.add(f"{a.chain.name}_{a.residue.seqid.num}_{a.residue.seqid.icode.strip()}")

            # store data
            xyz_l.append([a.atom.pos.x, a.atom.pos.y, a.atom.pos.z])
            element_l.append(a.atom.element.name)
            name_l.append(a.atom.name)
            resname_l.append(a.residue.name)
            resid_l.append(len(resid_set))
            chain_name_l.append(a.chain.name)

    # pack data
    return Structure(
        xyz=np.array(xyz_l, dtype=np.float32),
        names=np.array(name_l),
        elements=np.array(element_l),
        resnames=np.array(resname_l),
        resids=np.array(resid_l, dtype=np.int32),
        chain_names=np.array(chain_name_l),
    )


def subunit_to_pdb_str(subunit: Structure, chain_name, bfactors={}):
    # extract data
    pdb_str = ""
    for i in range(subunit.xyz.shape[0]):
        h = "ATOM" if subunit.resnames[i] in std_resnames else "HETATM"
        n = subunit.names[i]
        rn = subunit.resnames[i]
        e = subunit.elements[i]
        ri = subunit.resids[i]
        xyz = subunit.xyz[i]
        if chain_name in bfactors:
            bf = bfactors[chain_name][i]
        else:
            bf = 0.0

        # format pdb line
        pdb_str += (
            "{:<6s}{:>5d}  {:<4s}{:>3s} {:1s}{:>4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:<2s}  \n".format(
                h, i + 1, n, rn, chain_name, ri, xyz[0], xyz[1], xyz[2], 0.0, bf, e
            )
        )

    return pdb_str


def save_pdb(structure: Structure, filepath: str, bfactors={}):
    # split by chain
    subunits = split_by_chain(structure)

    # open file stream
    with open(filepath, "w") as fs:
        for cn in subunits:
            # get subunit
            subunit = subunits[cn]

            # convert subunit to string
            pdb_str = subunit_to_pdb_str(subunit, cn, bfactors)

            # write to file
            fs.write(pdb_str + "TER\n")

        # end of file
        fs.write("END")


def save_traj_pdb(structure: Structure, filepath):
    # split by chain
    subunits = split_by_chain(structure)
    xyz_dict = {cn: subunits[cn].xyz.copy() for cn in subunits}

    # determine number of frames
    assert len(structure.xyz.shape) == 3, "no time dimension"
    num_frames = structure.xyz.shape[0]

    # open file stream
    with open(filepath, "w") as fs:
        for k in range(num_frames):
            fs.write("MODEL    {:>4d}\n".format(k))
            for cn in subunits:
                # get subunit
                subunit = subunits[cn]

                # set coordinates to frame
                subunit.xyz = xyz_dict[cn][k]

                # convert subunit to string
                pdb_str = subunit_to_pdb_str(subunit, cn)

                # write to file
                fs.write(pdb_str + "\nTER\n")

            # end of model
            fs.write("ENDMDL\n")

        # end of file
        fs.write("END")


class StructuresDataset(pt.utils.data.Dataset):
    def __init__(self, pdb_filepaths, rm_wat=False, rm_clash=False):
        super(StructuresDataset).__init__()
        # store dataset filepath
        self.pdb_filepaths = pdb_filepaths

        # store flag
        self.rm_wat = rm_wat
        self.rm_clash = rm_clash

    def __len__(self):
        return len(self.pdb_filepaths)

    def __getitem__(self, i) -> tuple[Structure | None, str]:
        # find pdb filepath
        pdb_filepath = self.pdb_filepaths[i]

        # parse pdb
        structure = load_structure(pdb_filepath)
        return structure, pdb_filepath
