import numpy as np
import mrcfile as mf


def read_map(map_filepath):
    # load mrc file
    with mf.open(map_filepath, "r") as mrc:
        header = mrc.header
        data = mrc.data
        # mrc.print_header()

    # check data is available
    if (header is None) or (data is None):
        return {}

    if (header.nx != header.mx) or (header.ny != header.my) or (header.nz != header.mz):
        return {}

    # locate voxel centers
    hx = np.linspace(0.0, header.cella.x, header.nx + 1)[:-1]
    hx = hx + header.nxstart * header.cella.x / header.nx
    hy = np.linspace(0.0, header.cella.y, header.ny + 1)[:-1]
    hy = hy + header.nystart * header.cella.y / header.ny
    hz = np.linspace(0.0, header.cella.z, header.nz + 1)[:-1]
    hz = hz + header.nzstart * header.cella.z / header.nz

    # find axis order
    ids_axis = np.argsort(np.array([header.maps, header.mapr, header.mapc]))

    # print(data.shape, hx.shape, hy.shape, hz.shape, ids_axis)

    return {
        "H": np.transpose(data.copy(), ids_axis),
        "hx": hx,
        "hy": hy,
        "hz": hz,
    }


def crop_map(H, hx, hy, hz, xyz, dpad=5.0):
    # get box around structure
    xyz_min = np.min(xyz, axis=0) - dpad
    xyz_max = np.max(xyz, axis=0) + dpad

    # convert to map index
    mx = (xyz_min[0] <= hx) & (hx <= xyz_max[0])
    my = (xyz_min[1] <= hy) & (hy <= xyz_max[1])
    mz = (xyz_min[2] <= hz) & (hz <= xyz_max[2])

    if not (np.any(mx) and np.any(my) and np.any(mz)):
        return H, hx, hy, hz, xyz

    # crop map
    H = H[mx][:, my][:, :, mz]

    # crop map coordinates
    hx = hx[mx]
    hy = hy[my]
    hz = hz[mz]

    return H, hx, hy, hz, xyz
