import numpy as np
import torch as pt

from .standard_encoding import std_elements, std_element_radii


def superpose(xyz_ref: pt.Tensor, xyz: pt.Tensor) -> pt.Tensor:
    # centering
    t = pt.mean(xyz, dim=1).unsqueeze(1)
    t_ref = pt.mean(xyz_ref, dim=1).unsqueeze(1)

    # SVD decomposition
    U, _, Vt = pt.linalg.svd(pt.matmul(pt.transpose(xyz_ref - t_ref, 1, 2), xyz - t))

    # reflection matrix
    Z = pt.zeros(U.shape, device=xyz.device) + pt.eye(U.shape[1], U.shape[2], device=xyz.device).unsqueeze(0)
    Z[:, -1, -1] = pt.linalg.det(U) * pt.linalg.det(Vt)

    R = pt.matmul(pt.transpose(Vt, 1, 2), pt.matmul(Z, pt.transpose(U, 1, 2)))

    return pt.matmul(xyz - t, R) + t_ref


def extract_geometry(Xi: pt.Tensor, Xj: pt.Tensor):
    # compute displacement vectors
    R = Xj.unsqueeze(0) - Xi.unsqueeze(1)
    # compute distance matrix
    D = pt.norm(R, dim=2)
    # normalize displacement vectors
    R = R / (D.detach().unsqueeze(2) + (D < 1e-3).unsqueeze(2).float())
    return D, R


def extract_neighborhood(D, R, max_nn):
    # find geometry nearest neighbors
    num_nn = min(D.shape[0], max_nn)
    _, ids_nn = pt.topk(mask_diagonal(D), num_nn, dim=1, largest=False)
    D_nn = pt.gather(D, 1, ids_nn)
    R_nn = pt.gather(R, 1, ids_nn.unsqueeze(2).repeat_interleave(R.shape[2], dim=2))

    return ids_nn, D_nn, R_nn


def extract_connectivity(qe, D, alpha=1.2):
    elements = np.concatenate([std_elements, ["X"]])[pt.argmax(qe, dim=1).cpu().numpy()]
    radii = pt.from_numpy(np.array([std_element_radii[e.upper()] for e in elements])).to(D.device)
    return (
        (D < np.sqrt(alpha) * (radii.unsqueeze(0) + radii.unsqueeze(1)))
        & ~pt.eye(D.shape[0], device=D.device, dtype=pt.bool)
    ).float()


def extract_neighborhood_with_edges(D, R, C, max_nn):
    # virtual distances with bonded atoms at -1 to ensure in topk
    Dv = -C + (1.0 - C) * mask_diagonal(D)

    # find nearest neighbors with edges
    num_nn = min(D.shape[0], max_nn)
    Dv_nn, ids_nn = pt.topk(Dv, num_nn, dim=1, largest=False)
    D_nn = pt.gather(D, 1, ids_nn)
    R_nn = pt.gather(R, 1, ids_nn.unsqueeze(2).repeat((1, 1, R.shape[2])))
    E_nn = (Dv_nn < 0.0).float()

    return ids_nn, D_nn, R_nn, E_nn


def connected_distance_matrix(C):
    # initialize state
    S = C.clone()
    L = C.clone()
    E = pt.eye(C.shape[0], device=C.device)

    # iterate
    for i in range(C.shape[0]):
        # propagate information through graph
        S = pt.clip(pt.matmul(C, S), min=0.0, max=1.0)
        # deactivate already activated cells
        S = pt.clip(S - L - E, min=0.0, max=1.0)
        # update paths length
        L += (i + 2) * S

        # check convergence
        if pt.sum(S) == 0.0:
            break

    return L


def mask_diagonal(D):
    return D + 2.0 * pt.max(D.detach()) * pt.eye(D.shape[0], device=D.device)


def follow_rabbit(M, i):
    ids_checked = {i}
    ids_checking = set(np.where(M[i])[0])
    while ids_checking:
        for j in ids_checking.copy():
            ids_checking.remove(j)
            ids_checking.update(set([i for i in np.where(M[j])[0] if i not in ids_checked]))
            ids_checked.add(j)

    return list(ids_checked)


def follow_rabbits(M):
    i = 0
    ids_checked = []
    ids_clust = []
    while len(ids_checked) < M.shape[0]:
        ids_connect = follow_rabbit(M, i)
        ids_checked.extend(ids_connect)
        ids_clust.append(ids_connect)
        for j in range(i, M.shape[0]):
            if j not in ids_checked:
                i = j
                break

    return ids_clust


def find_bonded_graph_neighborhood(L, D, num_bond):
    # find neighborhood in bonded graph space
    D1 = L + 0.999 * (D.detach() / pt.max(D.detach())) + 2.0 * pt.max(L) * (L < 0.5).float()
    _, ids_nn = pt.topk(D1, num_bond, dim=1, largest=False)

    # map unbonded atoms to itself
    m_bb_nn = pt.gather(L < 1.0, 1, ids_nn)
    m_bb_nn = m_bb_nn & ~pt.all(m_bb_nn, dim=1).reshape(-1, 1)
    ids0, ids1 = pt.where(m_bb_nn)
    ids_nn[ids0, ids1] = ids0

    return ids_nn


def connected_paths(C, length):
    Mc = C > 0.5
    Gc = [np.array([])] + [np.where(Mc[i])[0] + 1 for i in range(Mc.shape[0])]
    cids = [[i] for i in range(len(Gc))]
    for n in range(length):
        cids_next = []
        for k in range(len(cids)):
            ids_next = Gc[cids[k][-1]]
            if len(ids_next) > 0:
                for i in ids_next:
                    if i not in cids[k]:
                        cids_next.extend([cids[k].copy() + [i]])
                    else:
                        cids_next.extend([cids[k].copy() + [0]])
            else:
                cids_next.extend([cids[k].copy() + [0]])
        cids = cids_next
    return np.array(cids) - 1


def topology_hash(C, qe, length):
    # get all connected paths up to length
    cpaths = connected_paths(C, length)

    # hash connections per atom
    # qs = np.array(["{}-{}-{}".format(ve,vr,vn) for ve,vr,vn in zip(np.argmax(qe, axis=1), np.argmax(qr, axis=1), np.argmax(qn, axis=1))])
    qs = np.array(["{}".format(ve) for ve in np.argmax(qe, axis=1)])
    hs = []
    for k in np.unique(cpaths[1:, 0]):
        cpk = cpaths[cpaths[:, 0] == k]
        hsk = []
        for i in range(cpk.shape[0]):
            hsi = []
            for j in range(cpk.shape[1]):
                if cpk[i, j] >= 0:
                    hsi.append(qs[cpk[i, j]])
                # else:
                # hsi.append('_')
            hsk.append(":".join(hsi))

        hs.append("+".join(sorted(hsk)))
    return np.array(hs)


def extract_context_map(C, qe, path_length=5):
    # hash topology and get indices of context
    hs = topology_hash(C.cpu().numpy(), qe.cpu().numpy(), path_length)
    hsu, ids_ctx = np.unique(hs, return_inverse=True)
    Mc = pt.stack([pt.from_numpy(ids_ctx == i) for i in range(hsu.shape[0])], dim=1)

    return Mc.to(C.device)


def locate_contacts(xyz_i, xyz_j, r_thr):
    with pt.no_grad():
        # compute distance matrix between subunits
        D = pt.norm(xyz_i.unsqueeze(1) - xyz_j.unsqueeze(0), dim=2)

        # find contacts
        ids_i, ids_j = pt.where(D < r_thr)

        # get contacts distances
        d_ij = D[ids_i, ids_j]

    return ids_i, ids_j, d_ij


def compute_rmsd(xyz0, xyz1):
    # superpose
    xyz1, xyz0 = superpose(xyz0.view(1, -1, 3), xyz1.view(1, -1, 3))

    # compute rmsd
    rmsd = pt.sqrt(pt.mean(pt.sum(pt.square(xyz0 - xyz1), dim=2)))

    return rmsd


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
