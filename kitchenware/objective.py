import torch as pt

from scipy.optimize import linear_sum_assignment


def compute_distance_matrix(X):
    return pt.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=2)


def superpose_transform(X0, X):
    # weighted coordinate centering
    t0 = pt.mean(X0, dim=0).unsqueeze(0)
    t = pt.mean(X, dim=0).unsqueeze(0)
    X0c = X0 - t0
    Xc = X - t

    # SVD decomposition
    B = pt.matmul(pt.transpose(X0c, 0, 1), Xc) / X0c.shape[0]
    U, _, Vt = pt.linalg.svd(B)

    # reflection matrix
    Z = pt.zeros(U.shape, device=X0c.device) + pt.eye(U.shape[0], U.shape[1], device=X0c.device)
    Z[-1, -1] = pt.linalg.det(U) * pt.linalg.det(Vt)

    # rotation matrix
    R = pt.matmul(Vt.T, pt.matmul(Z, U.T))

    return R, t, t0


def superpose(X0, X):
    # weighted coordinate centering
    t0 = pt.mean(X0, dim=0).unsqueeze(0)
    t = pt.mean(X, dim=0).unsqueeze(0)
    X0c = X0 - t0
    Xc = X - t

    # SVD decomposition
    B = pt.matmul(pt.transpose(X0c, 0, 1), Xc) / X0c.shape[0]
    U, _, Vt = pt.linalg.svd(B)

    # reflection matrix
    Z = pt.zeros(U.shape, device=X0c.device) + pt.eye(U.shape[0], U.shape[1], device=X0c.device)
    Z[-1, -1] = pt.linalg.det(U) * pt.linalg.det(Vt)

    # rotation matrix
    R = pt.matmul(Vt.T, pt.matmul(Z, U.T))

    return pt.matmul(X - t, R) + t0


def reduce_tags(q, M):
    return pt.matmul(M.T, q)


def tags_scales(q, M_l):
    # get target reduced coordinates
    q_l = [q]
    for M in M_l:
        q_l.append(reduce_tags(q_l[-1], M))

    return q_l


def extract_context_map(C0, qe):
    ids_ctx = pt.unique(pt.sum((C0.unsqueeze(2) * qe.unsqueeze(0)), dim=1), return_inverse=True, dim=0)[1]
    K = ids_ctx.reshape(-1, 1) == pt.unique(ids_ctx).reshape(1, -1)
    return K


def optimal_permutation(X, X0, K):
    idsp = pt.arange(K.shape[0], dtype=pt.long, device=K.device)
    for i in range(K.shape[1]):
        mki = K[:, i]
        Dk = pt.norm(X[mki].unsqueeze(1) - X0[mki].unsqueeze(0), dim=2)
        idskp = pt.from_numpy(linear_sum_assignment(Dk.cpu().numpy())[1]).to(Dk.device)

        idsc = pt.where(mki)[0]
        idsp[idsc] = idsc[idskp]

    return idsp


def weighted_superpose(X0, X, w):
    # normalize weight
    w = (w / pt.sum(w, dim=1).unsqueeze(1)).unsqueeze(2)

    # weighted coordinate centering
    t0 = pt.sum(w * X0, dim=1).unsqueeze(1)
    t = pt.sum(w * X, dim=1).unsqueeze(1)
    X0c = X0 - t0
    Xc = X - t

    # SVD decomposition
    B = pt.matmul(pt.transpose(X0c, 1, 2), w * Xc) / X0c.shape[1]
    U, _, Vt = pt.linalg.svd(B)

    # reflection matrix
    Z = pt.zeros(U.shape).to(X0c.device) + pt.eye(U.shape[1], U.shape[2]).to(X0c.device).unsqueeze(0)
    Z[:, -1, -1] = pt.linalg.det(U) * pt.linalg.det(Vt)

    # rotation matrix
    R = pt.matmul(pt.transpose(Vt, 1, 2), pt.matmul(Z, pt.transpose(U, 1, 2)))

    return pt.matmul(X - t, R) + t0


def multiple_superpose(X0, X):
    # weighted coordinate centering
    t0 = pt.sum(X0, dim=1).unsqueeze(1)
    t = pt.sum(X, dim=1).unsqueeze(1)
    X0c = X0 - t0
    Xc = X - t

    # SVD decomposition
    B = pt.matmul(pt.transpose(X0c, 1, 2), Xc) / X0c.shape[1]
    U, _, Vt = pt.linalg.svd(B)

    # reflection matrix
    Z = pt.zeros(U.shape).to(X0c.device) + pt.eye(U.shape[1], U.shape[2]).to(X0c.device).unsqueeze(0)
    Z[:, -1, -1] = pt.linalg.det(U) * pt.linalg.det(Vt)

    # rotation matrix
    R = pt.matmul(pt.transpose(Vt, 1, 2), pt.matmul(Z, pt.transpose(U, 1, 2)))

    return pt.matmul(X - t, R) + t0


def local_alignment_error(X, X0, C0, num_nn=4):
    # compute distance matrix
    D = pt.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=2)
    # D0 = pt.norm(X0.unsqueeze(0) - X0.unsqueeze(1), dim=2)

    # virtual distances with bonded atoms at -1 to ensure in topk
    Dv = -C0 + (1.0 - C0) * D

    # find nearest neighbors with edges
    _, ids_nn = pt.topk(Dv, num_nn, dim=1, largest=False)
    D_nn = pt.gather(D.detach(), 1, ids_nn)
    # D0_nn = pt.gather(D0, 1, ids_nn)

    # extract neighborhoods subsets
    X_nn = X[ids_nn]
    X0_nn = X0[ids_nn]

    # superposition weight based on distance
    w = 1.0 / (1.0 + pt.square(D_nn))
    # w = pt.exp(-pt.abs(D_nn - D0_nn) / (D0_nn + 1e-3))

    # weighted superpose
    X0_nn = weighted_superpose(X_nn, X0_nn, w)
    # X0_nn = multiple_superpose(X_nn, X0_nn)

    # compute error
    r_nn = pt.norm(X_nn - X0_nn, dim=2)

    return pt.mean(r_nn, dim=1)


def multiscale_fragments_losses(X_l, X0_l, C_l):
    losses = []
    for i in range(len(X_l)):
        if X_l[i].shape[0] > 1:
            # local alignment error
            num_nn = min(4, X_l[i].shape[0])
            lsi = local_alignment_error(X_l[i], X0_l[i], C_l[i], num_nn=num_nn)

            # global alignment
            Xgi = superpose(X_l[i], X0_l[i])
            lgi = pt.norm(X_l[i] - Xgi, dim=1)

            # loss scale
            gamma = (X0_l[0].shape[0] / X0_l[i].shape[0]) ** 0.5

            # combined loss
            losses.append(pt.mean(lsi * lgi) * gamma)
            # losses.append(pt.mean(lgi))
            # losses.append(pt.mean(lsi))

    losses = pt.sqrt(pt.stack(losses))

    return losses


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
