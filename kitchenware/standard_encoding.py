import numpy as np


# standard elements (sorted by aboundance) (32)
std_elements = np.array(
    [
        "C",
        "O",
        "N",
        "S",
        "P",
        "Se",
        "Mg",
        "Cl",
        "Zn",
        "Fe",
        "Ca",
        "Na",
        "F",
        "Mn",
        "I",
        "K",
        "Br",
        "Cu",
        "Cd",
        "Ni",
        "Co",
        "Sr",
        "Hg",
        "W",
        "As",
        "B",
        "Mo",
        "Ba",
        "Pt",
    ]
)

# standard residue names: AA/RNA/DNA (sorted by aboundance) (29)
std_resnames = np.array(
    [
        "LEU",
        "GLU",
        "ARG",
        "LYS",
        "VAL",
        "ILE",
        "PHE",
        "ASP",
        "TYR",
        "ALA",
        "THR",
        "SER",
        "GLN",
        "ASN",
        "PRO",
        "GLY",
        "HIS",
        "TRP",
        "MET",
        "CYS",
        "G",
        "A",
        "C",
        "U",
        "DG",
        "DA",
        "DT",
        "DC",
    ]
)

# standard atom names contained in standard residues (sorted by aboundance) (63)
std_names = np.array(
    [
        "CA",
        "N",
        "C",
        "O",
        "CB",
        "CG",
        "CD2",
        "CD1",
        "CG1",
        "CG2",
        "CD",
        "OE1",
        "OE2",
        "OG",
        "OG1",
        "OD1",
        "OD2",
        "CE",
        "NZ",
        "NE",
        "CZ",
        "NH2",
        "NH1",
        "ND2",
        "CE2",
        "CE1",
        "NE2",
        "OH",
        "ND1",
        "SD",
        "SG",
        "NE1",
        "CE3",
        "CZ3",
        "CZ2",
        "CH2",
        "P",
        "C3'",
        "C4'",
        "O3'",
        "C5'",
        "O5'",
        "O4'",
        "C1'",
        "C2'",
        "O2'",
        "OP1",
        "OP2",
        "N9",
        "N2",
        "O6",
        "N7",
        "C8",
        "N1",
        "N3",
        "C2",
        "C4",
        "C6",
        "C5",
        "N6",
        "N4",
        "O2",
        "O4",
    ]
)

# backbone
std_backbone = np.array(
    [
        "CA",
        "N",
        "C",
        "O",
        # "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'",
        # "C3'", "O3'", "C2'", "O2'", "C1'",
    ]
)

# amino-acids
std_aminoacids = np.array(
    [
        "LEU",
        "GLU",
        "ARG",
        "LYS",
        "VAL",
        "ILE",
        "PHE",
        "ASP",
        "TYR",
        "ALA",
        "THR",
        "SER",
        "GLN",
        "ASN",
        "PRO",
        "GLY",
        "HIS",
        "TRP",
        "MET",
        "CYS",
    ]
)

# resname categories
categ_to_resnames = {
    "protein": [
        "GLU",
        "LEU",
        "ALA",
        "ASP",
        "SER",
        "VAL",
        "GLY",
        "THR",
        "ARG",
        "PHE",
        "TYR",
        "ILE",
        "PRO",
        "ASN",
        "LYS",
        "GLN",
        "HIS",
        "TRP",
        "MET",
        "CYS",
    ],
    "rna": ["A", "U", "G", "C"],
    "dna": ["DA", "DT", "DG", "DC"],
    "ion": ["MG", "ZN", "CL", "CA", "NA", "MN", "K", "IOD", "CD", "CU", "FE", "NI", "SR", "BR", "CO", "HG"],
    "ligand": [
        "SO4",
        "NAG",
        "PO4",
        "EDO",
        "ACT",
        "MAN",
        "HEM",
        "FMT",
        "BMA",
        "ADP",
        "FAD",
        "NAD",
        "NO3",
        "GLC",
        "ATP",
        "NAP",
        "BGC",
        "GDP",
        "FUC",
        "FES",
        "FMN",
        "GAL",
        "GTP",
        "PLP",
        "MLI",
        "ANP",
        "H4B",
        "AMP",
        "NDP",
        "SAH",
        "OXY",
    ],
    "lipid": ["PLM", "CLR", "CDL", "RET"],
}
resname_to_categ = {rn: c for c in categ_to_resnames for rn in categ_to_resnames[c]}

std_element_radii = {
    "H": 0.58,
    "HE": 0.441,
    "LI": 1.89,
    "BE": 1.26,
    "B": 1.08,
    "C": 0.73,
    "N": 0.75,
    "O": 0.73,
    "F": 0.71,
    "NE": 0.459,
    "NA": 0.75,
    "MG": 1.53,
    "AL": 1.62,
    "SI": 1.35,
    "P": 1.06,
    "S": 1.02,
    "CL": 0.99,
    "AR": 0.792,
    "K": 2.52,
    "CA": 1.5,
    "SC": 1.89,
    "TI": 1.8,
    "V": 1.71,
    "CR": 1.2,
    "MN": 1.62,
    "FE": 1.55,
    "CO": 1.53,
    "NI": 1.44,
    "CU": 1.44,
    "ZN": 1.35,
    "GA": 1.62,
    "GE": 1.35,
    "AS": 1.17,
    "SE": 1.08,
    "BR": 1.14,
    "KR": 0.9,
    "RB": 2.7,
    "SR": 2.25,
    "Y": 2.07,
    "ZR": 1.98,
    "NB": 1.89,
    "MO": 1.8,
    "TC": 1.8,
    "RU": 1.71,
    "RH": 1.62,
    "PD": 1.62,
    "AG": 1.62,
    "CD": 1.53,
    "IN": 1.8,
    "SN": 1.53,
    "SB": 1.35,
    "TE": 1.26,
    "I": 1.33,
    "XE": 1.08,
    "CS": 2.97,
    "BA": 2.52,
    "LA": 2.43,
    "CE": 2.43,
    "PR": 2.43,
    "ND": 2.34,
    "PM": 2.34,
    "SM": 2.34,
    "EU": 2.34,
    "GD": 2.25,
    "TB": 2.25,
    "DY": 2.25,
    "HO": 2.25,
    "ER": 2.25,
    "TM": 2.16,
    "YB": 2.16,
    "LU": 2.07,
    "HF": 1.98,
    "TA": 1.89,
    "W": 1.8,
    "RE": 1.8,
    "OS": 1.71,
    "IR": 1.71,
    "PT": 1.62,
    "AU": 1.44,
    "HG": 1.62,
    "TL": 1.89,
    "PB": 2,
    "BI": 2,
    "PO": 2,
    "AT": 2,
    "RN": 2,
    "FR": 2,
    "RA": 2,
    "AC": 2,
    "TH": 2,
    "PA": 2,
    "U": 2,
    "NP": 2,
    "PU": 2,
    "AM": 2,
    "CM": 2,
    "BK": 2,
    "CF": 2,
    "ES": 2,
    "FM": 2,
    "MD": 2,
    "NO": 2,
    "LR": 2,
    "RF": 2,
    "DB": 2,
    "SG": 2,
    "BH": 2,
    "HS": 2,
    "MT": 2,
    "DS": 2,
    "RG": 2,
    "CN": 2,
    "UUT": 2,
    "UUQ": 2,
    "UUP": 2,
    "UUH": 2,
    "UUS": 2,
    "UUO": 2,
    "X": 2,
}

# resname convergion (37)
res3to1 = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}
res1to3 = {v: k for k, v in res3to1.items()}