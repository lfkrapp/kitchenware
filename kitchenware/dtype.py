import numpy as np
import torch as pt
import numpy.typing as npt
from typing import Generator
from dataclasses import dataclass, fields


@dataclass
class Structure:
    xyz: npt.NDArray[np.float32]
    names: npt.NDArray[np.str_]
    elements: npt.NDArray[np.str_]
    resnames: npt.NDArray[np.str_]
    resids: npt.NDArray[np.int32]
    chain_names: npt.NDArray[np.str_]

    def __iter__(self):
        for field in fields(self):
            yield field.name

    def __getitem__(self, idx):
        return Structure(**{key: getattr(self, key)[idx] for key in self})


@dataclass
class StructureData:
    X: pt.Tensor
    qe: pt.Tensor
    qr: pt.Tensor
    qn: pt.Tensor
    Mr: pt.Tensor
    Mc: pt.Tensor

    def to(self, device):
        return StructureData(
            X=self.X.to(device),
            qe=self.qe.to(device),
            qr=self.qr.to(device),
            qn=self.qn.to(device),
            Mr=self.Mr.to(device),
            Mc=self.Mc.to(device),
        )

    def cpu(self):
        self.to(pt.device("cpu"))

    def __getitem__(self, idx):
        return StructureData(
            X=self.X[idx],
            qe=self.qe[idx],
            qr=self.qr[idx],
            qn=self.qn[idx],
            Mr=self.Mr[idx][:, pt.sum(self.Mr[idx], dim=0) > 0.5],
            Mc=self.Mc[idx][:, pt.sum(self.Mc[idx], dim=0) > 0.5],
        )

    def __iter__(self) -> Generator[pt.Tensor, None, None]:
        for field in fields(self):
            yield getattr(self, field.name)
