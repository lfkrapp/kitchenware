import torch as pt
from typing import List

from .geometry import superpose_many, superpose


def compute_rmsd_all(xyz0, xyz1):
    # superpose
    xyz1, xyz0 = superpose_many(xyz0.view(1, -1, 3), xyz1.view(1, -1, 3))

    # compute rmsd
    rmsd = pt.sqrt(pt.mean(pt.sum(pt.square(xyz0 - xyz1), dim=2)))

    return rmsd


def compute_rmsd(X0, X1):
    # superpose
    X1 = superpose(X0, X1)

    # compute rmsd
    rmsd = pt.sqrt(pt.mean(pt.sum(pt.square(X0 - X1), dim=1)))

    return rmsd


def compute_lDDT(X, X0, r_thr=[0.5, 1.0, 2.0, 4.0], R0=15.0):
    # compute distance matrices
    D = pt.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=2)
    D0 = pt.norm(X0.unsqueeze(0) - X0.unsqueeze(1), dim=2)

    # thresholds
    r_thr = pt.tensor(r_thr, device=D.device)

    # local selection mask
    M = ((D0 < R0) & (D0 > 0.0)).float()

    # compute score Local Distance Difference Test
    DD = (pt.abs(D0 - D).unsqueeze(0) < r_thr.view(-1, 1, 1)).float()
    lDD = pt.sum(DD * M.unsqueeze(0), dim=2) / pt.sum(M, dim=1).unsqueeze(0)
    lDDT = 1e2 * pt.mean(lDD, dim=0)

    return lDDT


def compute_gdt_ts(xyz0: pt.Tensor, xyz1: pt.Tensor, r_thr: List[float] = [1.0, 2.0, 4.0, 8.0]) -> float:
    # superpose
    xyz1_aligned, xyz0_aligned = superpose(xyz0.view(1, -1, 3), xyz1.view(1, -1, 3))

    # compute pairwise distances
    distances = pt.sqrt(pt.sum((xyz0_aligned - xyz1_aligned) ** 2, dim=2)).squeeze()

    # percentage of atoms within each threshold
    scores: List[float] = []
    for threshold in r_thr:
        score = pt.sum(distances < threshold).item() / distances.shape[0] * 100
        scores.append(score)

    # compute GDT_TS as the average of scores at each threshold
    gdt_ts = sum(scores) / len(r_thr)
    return gdt_ts


def angle(p1, p2, p3):
    # displacement vectors
    v1 = p2 - p1
    v2 = p2 - p3
    # normalize vectors
    v1 = v1 / pt.norm(v1, dim=1).unsqueeze(1)
    v2 = v2 / pt.norm(v2, dim=1).unsqueeze(1)
    # angle
    return pt.arccos(pt.sum(v1 * v2, dim=1))


def dihedral(p1, p2, p3, p4):
    # displacement vectors
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    # normalize vectors
    v1 = v1 / pt.norm(v1, dim=1).unsqueeze(1)
    v2 = v2 / pt.norm(v2, dim=1).unsqueeze(1)
    v3 = v3 / pt.norm(v3, dim=1).unsqueeze(1)
    # cross vectors
    r1 = pt.cross(v1, v2, dim=1)
    r2 = pt.cross(v2, v3, dim=1)
    # angle
    x = pt.sum(r1 * r2, dim=1)
    y = pt.sum(pt.cross(r1, r2, dim=1) * v2, dim=1)
    return pt.atan2(y, x)


def secondary_structure(ca_xyz):
    # constants
    _radians_to_angle = 2 * pt.pi / 360

    _r_helix = ((89 - 12) * _radians_to_angle, (89 + 12) * _radians_to_angle)
    _a_helix = ((50 - 20) * _radians_to_angle, (50 + 20) * _radians_to_angle)
    # _d2_helix = ((5.5-0.5), (5.5+0.5))
    _d3_helix = ((5.3 - 0.5), (5.3 + 0.5))
    _d4_helix = ((6.4 - 0.6), (6.4 + 0.6))

    _r_strand = ((124 - 14) * _radians_to_angle, (124 + 14) * _radians_to_angle)
    _a_strand = (
        (-180) * _radians_to_angle,
        (-125) * _radians_to_angle,
        (145) * _radians_to_angle,
        (180) * _radians_to_angle,
    )
    _d2_strand = ((6.7 - 0.6), (6.7 + 0.6))
    _d3_strand = ((9.9 - 0.9), (9.9 + 0.9))
    _d4_strand = ((12.4 - 1.1), (12.4 + 1.1))

    # define distances and angles buffers
    d2i_xyz = pt.full((ca_xyz.shape[0], 2, 3), pt.nan, device=ca_xyz.device)
    d3i_xyz = pt.full((ca_xyz.shape[0], 2, 3), pt.nan, device=ca_xyz.device)
    d4i_xyz = pt.full((ca_xyz.shape[0], 2, 3), pt.nan, device=ca_xyz.device)
    ri_xyz = pt.full((ca_xyz.shape[0], 3, 3), pt.nan, device=ca_xyz.device)
    ai_xyz = pt.full((ca_xyz.shape[0], 4, 3), pt.nan, device=ca_xyz.device)

    # fill distances and angles buffers
    d2i_xyz[1:-1] = pt.stack([ca_xyz[:-2], ca_xyz[2:]], dim=1)
    d3i_xyz[1:-2] = pt.stack([ca_xyz[:-3], ca_xyz[3:]], dim=1)
    d4i_xyz[1:-3] = pt.stack([ca_xyz[:-4], ca_xyz[4:]], dim=1)
    ri_xyz[1:-1] = pt.stack([ca_xyz[:-2], ca_xyz[1:-1], ca_xyz[2:]], dim=1)
    ai_xyz[1:-2] = pt.stack([ca_xyz[:-3], ca_xyz[1:-2], ca_xyz[2:-1], ca_xyz[3:]], dim=1)

    # compute distances and angles
    d2i = pt.linalg.norm(d2i_xyz[:, 0] - d2i_xyz[:, 1], dim=1)
    d3i = pt.linalg.norm(d3i_xyz[:, 0] - d3i_xyz[:, 1], dim=1)
    d4i = pt.linalg.norm(d4i_xyz[:, 0] - d4i_xyz[:, 1], dim=1)
    ri = angle(ri_xyz[:, 0], ri_xyz[:, 1], ri_xyz[:, 2])
    ai = dihedral(ai_xyz[:, 0], ai_xyz[:, 1], ai_xyz[:, 2], ai_xyz[:, 3])

    # initial secondary structure
    sse = pt.zeros(ca_xyz.shape[0], dtype=pt.long, device=ca_xyz.device)

    # potential helices
    c1 = (d3i >= _d3_helix[0]) & (d3i <= _d3_helix[1]) & (d4i >= _d4_helix[0]) & (d4i <= _d4_helix[1])
    c2 = (ri >= _r_helix[0]) & (ri <= _r_helix[1]) & (ai >= _a_helix[0]) & (ai <= _a_helix[1])
    is_pot_helix = c1 | c2

    # find helices
    cl = pt.conv1d(is_pot_helix.float().reshape(1, 1, -1), pt.ones(1, 1, 5, device=ca_xyz.device) / 5.0, padding="same")
    is_helix = (pt.max_pool1d(cl, 5, stride=1, padding=2).floor() > 0.5).squeeze()

    # extend helices backward
    c1 = (d3i[:-1] >= _d3_helix[0]) & (d3i[:-1] <= _d3_helix[1])
    c2 = (ri[:-1] >= _r_helix[0]) & (ri[:-1] <= _r_helix[1])
    is_helix[:-1] = is_helix[:-1] | ((c1 | c2) & is_helix[1:])

    # extend helices forward
    c1 = (d3i[1:] >= _d3_helix[0]) & (d3i[1:] <= _d3_helix[1])
    c2 = (ri[1:] >= _r_helix[0]) & (ri[1:] <= _r_helix[1])
    is_helix[1:] = is_helix[1:] | ((c1 | c2) & is_helix[:-1])

    # update sse with helices
    sse[is_helix] = 1

    # potential strands
    c1 = (d2i >= _d2_strand[0]) & (d2i <= _d2_strand[1])
    c2 = (d3i >= _d3_strand[0]) & (d3i <= _d3_strand[1])
    c3 = (d4i >= _d4_strand[0]) & (d4i <= _d4_strand[1])
    c4 = (ri >= _r_strand[0]) & (ri <= _r_strand[1])
    c5 = ((ai >= _a_strand[0]) & (ai <= _a_strand[1])) | ((ai >= _a_strand[2]) & (ai <= _a_strand[3]))
    is_pot_strand = (c1 & c2 & c3) | (c4 & c5)

    # strand is long enough
    cl = pt.conv1d(
        is_pot_strand.float().reshape(1, 1, -1), pt.ones(1, 1, 5, device=ca_xyz.device) / 5.0, padding="same"
    )
    is_strand_c1 = (pt.max_pool1d(cl, 5, stride=1, padding=2).floor() > 0.5).squeeze()

    # strand has enough neighboring strands
    D_ca = pt.norm(ca_xyz.unsqueeze(0) - ca_xyz.unsqueeze(1), dim=2)
    C_strand = (is_pot_strand.unsqueeze(0) & is_pot_strand.unsqueeze(1)) & ((D_ca >= 4.2) & (D_ca <= 5.2))
    c_strand = pt.sum(C_strand.float(), dim=0)
    cc = pt.conv1d(c_strand.reshape(1, 1, -1), pt.ones(1, 1, 3, device=ca_xyz.device) / 5.0, padding="same")
    is_strand_c2 = (pt.max_pool1d(cc, 3, stride=1, padding=1).floor() > 0.5).squeeze()

    # strands
    is_strand = is_strand_c1 | is_strand_c2

    # extend strands forward
    c1 = (d3i[:-1] >= _d3_strand[0]) & (d3i[:-1] <= _d3_strand[1])
    is_strand[:-1] = is_strand[:-1] | (c1 & is_strand[1:])

    # extrand strands backward
    c2 = (d3i[1:] >= _d3_strand[0]) & (d3i[1:] <= _d3_strand[1])
    is_strand[1:] = is_strand[1:] | (c2 & is_strand[:-1])

    # update sse with strands
    sse[is_strand] = 2

    return sse
