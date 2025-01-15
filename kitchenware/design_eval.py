from copy import deepcopy
import gzip
import math
import os.path as op

import _pickle as cPickle
import numpy as np

from rdkit import Chem
from rdkit.Chem import Crippen, DataStructs, Descriptors, Lipinski, rdmolfiles
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import *



# Function to convert the ligand PDB to SMILES using RDKit
def pdb_to_smiles(pdb_filename):
    mol = rdmolfiles.MolFromPDBFile(pdb_filename)

    if mol:
        return Chem.MolToSmiles(mol)
    else:
        return None

# Function to extract the ligand and save it as a new PDB file
def extract_ligand(pdb_data, ligand_name, output_pdb_filename):
    # Find the indices of atoms that belong to the ligand (based on residue name)
    ligand_indices = np.where(pdb_data['resname'] == ligand_name)[0]

    # Extract relevant data for the ligand
    ligand_xyz = pdb_data['xyz'][ligand_indices]
    ligand_atom_names = pdb_data['name'][ligand_indices]
    ligand_elements = pdb_data['element'][ligand_indices]

    # Write the ligand data to a new PDB file
    with open(output_pdb_filename, 'w') as f:
        for i, (xyz, name, element) in enumerate(zip(ligand_xyz, ligand_atom_names, ligand_elements)):
            # Ensure the element is formatted properly (e.g., 'C' for Carbon, 'N' for Nitrogen)
            element = element.capitalize()

            # Write the atom line in PDB format
            f.write(f"HETATM  {i+1:5d} {name:<4} {ligand_name:3}  1    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00           {element}\n")
        f.write("END\n")

def tanimoto_sim_smiles(smiles_mol, smiles_ref):
    mol = Chem.MolFromSmiles(smiles_mol)
    ref_mol = Chem.MolFromSmiles(smiles_ref)
    fp1 = Chem.RDKFingerprint(mol)
    fp2 = Chem.RDKFingerprint(ref_mol)
    return DataStructs.TanimotoSimilarity(fp1, fp2) 


def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp := Crippen.MolLogP(mol) >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#


_fscores = None


def readFragmentScores(name='fpscores'):
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    _fscores = cPickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,
                                               2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    # for bitId, v in iteritems(fps):
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore




#
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
#
#     * Redistributions of source code must retain the above copyright 
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following 
#       disclaimer in the documentation and/or other materials provided 
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc. 
#       nor the names of its contributors may be used to endorse or promote 
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

def compute_sa_score(rdmol):
    # rdmol = Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))
    sa = calculateScore(rdmol)
    sa_norm = round((10 - sa) / 9, 2)
    return sa, sa_norm
