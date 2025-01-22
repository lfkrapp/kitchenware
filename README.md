# Kitchenware

Utility tools to load and process biomolecular data. This package provides a foundation for deep learning applications in structural biology, offering efficient tools for handling molecular structures and their analysis.

## Features

**Data Processing**
- Load and save molecular structures (PDB/CIF formats)
- Structure encoding and processing
- Biomolecular data type handling

**Geometric Analysis**
- Distance matrix computation
- Connectivity extraction
- Neighborhood analysis
- Structure superposition
- Contact detection

**Structure Analysis**
- Secondary structure analysis
- RMSD calculations
- lDDT scoring
- Angle and dihedral measurements

**Utilities**
- Standard amino acid encodings
- Template handling
- Chain and residue operations
- Density map operations

## Installation

```bash
pip install kitchenware
```

## Quick Start

```python
import kitchenware as kw

# Load a structure (PDB or CIF format)
structure = kw.load_structure("input_structure.pdb")

# Encode structure and convert to a collection of tensors (PyTorch)
data = kw.encode_structure(structure)

# Convert back to a structure
structure = kw.data_to_structure(data)

# Write PDB files
kw.save_pdb(data, "output_structure.pdb")
```

## Dependencies

- gemmi
- mrcfile
- numpy
- pandas
- torch
- rdkit
- scipy
- scikit-learn

For a list of dependencies see [pyproject.toml](./pyproject.toml)

## Applications

This toolkit serves as the foundation for several deep learning methods in structural biology:
- [PeSTo](https://github.com/LBM-EPFL/PeSTo): parameter-free geometric deep learning for accurate prediction of protein binding interfaces
- [CARBonAra](https://github.com/LBM-EPFL/CARBonAra): Context-aware geometric deep learning for protein sequence design

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Reference

This toolkit was developed as the foundation for both PeSTo and CARBonAra. If you use this toolkit in your research, please cite one of the following publications.

### PeSTo

Krapp, L.F., Abriata, L.A., Cortés Rodriguez, F. et al. PeSTo: parameter-free geometric deep learning for accurate prediction of protein binding interfaces. Nat Commun 14, 2175 (2023). https://doi.org/10.1038/s41467-023-37701-8

### CARBonAra

Krapp, L.F., Meireles, F.A., Abriata, L.A. et al. Context-aware geometric deep learning for protein sequence design. Nat Commun 15, 6273 (2024). https://doi.org/10.1038/s41467-024-50571-y
