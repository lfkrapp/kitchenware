import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, fields


@dataclass
class Structure:
    xyz: npt.NDArray[np.float32]
    names: npt.NDArray[np.str_]
    elements: npt.NDArray[np.str_]
    resnames: npt.NDArray[np.str_]
    resids: npt.NDArray[np.int32]
    het_flags: npt.NDArray[np.str_]
    chain_names: npt.NDArray[np.str_]
    icodes: npt.NDArray[np.str_]
    bfactors: npt.NDArray[np.float32]

    def __iter__(self):
        for field in fields(self):
            yield field.name

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)
