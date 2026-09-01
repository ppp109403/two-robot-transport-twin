"""OccupancyGrid -> signed distance field with a CasADi-differentiable query.

The SDF is positive in free space and negative inside obstacles, in metres.
Keeping it signed (rather than clamping at zero) matters for IPOPT: if the
initial guess pokes into a wall the gradient still points back out instead of
vanishing.

Internally the grid is stored as ``sdf[ix, iy]`` (x-major).  That is the layout
CasADi's ``interpolant`` expects once flattened in Fortran order, so the query
function and the numpy sampler below stay in agreement by construction.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


class SignedDistanceField:
    def __init__(self, sdf_xy: np.ndarray, origin, res: float):
        self.sdf = np.ascontiguousarray(sdf_xy, dtype=np.float64)
        self.origin = np.asarray(origin, dtype=np.float64).reshape(2)
        self.res = float(res)
        self.nx, self.ny = self.sdf.shape
        # cell centres
        self.xs = self.origin[0] + (np.arange(self.nx) + 0.5) * self.res
        self.ys = self.origin[1] + (np.arange(self.ny) + 0.5) * self.res

    # ------------------------------------------------------------------ build
    @classmethod
    def from_occupancy(cls, data_yx, origin, res,
                       occupied_threshold: int = 50,
                       unknown_is_occupied: bool = True,
                       smooth_sigma_cells: float = 0.0):
        """``data_yx`` is the OccupancyGrid data reshaped to (height, width)."""
        data = np.asarray(data_yx)
        occ = data >= occupied_threshold
        if unknown_is_occupied:
            occ = occ | (data < 0)
        free = ~occ

        if not occ.any():
            sdf_yx = np.full(data.shape, 1.0e3, dtype=np.float64)
        elif not free.any():
            sdf_yx = np.full(data.shape, -1.0e3, dtype=np.float64)
        else:
            d_free = distance_transform_edt(free) * res
            d_occ = distance_transform_edt(occ) * res
            sdf_yx = d_free - d_occ

        if smooth_sigma_cells > 0.0:
            sdf_yx = gaussian_filter(sdf_yx, smooth_sigma_cells, mode='nearest')

        return cls(sdf_yx.T, origin, res)      # (ny, nx) -> (nx, ny)

    # ------------------------------------------------------------------- crop
    def crop(self, pts_xy, margin: float) -> 'SignedDistanceField':
        """Restrict the field to the bounding box of ``pts_xy`` plus ``margin``.

        The optimizer only ever queries near the reference path, and shrinking
        the grid is what keeps the b-spline interpolant construction cheap.
        """
        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
        lo = pts.min(axis=0) - margin
        hi = pts.max(axis=0) + margin

        ix0 = int(np.clip(np.floor((lo[0] - self.origin[0]) / self.res), 0, self.nx - 1))
        ix1 = int(np.clip(np.ceil((hi[0] - self.origin[0]) / self.res), 1, self.nx))
        iy0 = int(np.clip(np.floor((lo[1] - self.origin[1]) / self.res), 0, self.ny - 1))
        iy1 = int(np.clip(np.ceil((hi[1] - self.origin[1]) / self.res), 1, self.ny))

        # b-spline interpolation needs a few cells of slack on every side
        ix0 = max(0, ix0 - 4); iy0 = max(0, iy0 - 4)
        ix1 = min(self.nx, ix1 + 4); iy1 = min(self.ny, iy1 + 4)

        new_origin = self.origin + np.array([ix0, iy0]) * self.res
        return SignedDistanceField(self.sdf[ix0:ix1, iy0:iy1], new_origin, self.res)

    def downsample(self, factor: int) -> 'SignedDistanceField':
        if factor <= 1:
            return self
        # min-pool: keeps the conservative (closest-to-obstacle) value
        nx = (self.nx // factor) * factor
        ny = (self.ny // factor) * factor
        block = self.sdf[:nx, :ny].reshape(nx // factor, factor, ny // factor, factor)
        return SignedDistanceField(block.min(axis=(1, 3)),
                                   self.origin + 0.5 * (factor - 1) * self.res,
                                   self.res * factor)

    # ------------------------------------------------------------------ query
    def casadi_interpolant(self, name: str = 'sdf', kind: str = 'bspline'):
        import casadi as ca
        vals = self.sdf.ravel(order='F')          # index = ix + nx*iy
        return ca.interpolant(name, kind,
                              [self.xs.tolist(), self.ys.tolist()],
                              vals.tolist())

    def value(self, pts_xy) -> np.ndarray:
        """Bilinear sample, for warm-start checks and post-solve verification."""
        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
        fx = (pts[:, 0] - self.xs[0]) / self.res
        fy = (pts[:, 1] - self.ys[0]) / self.res
        ix = np.clip(np.floor(fx).astype(int), 0, self.nx - 2)
        iy = np.clip(np.floor(fy).astype(int), 0, self.ny - 2)
        tx = np.clip(fx - ix, 0.0, 1.0)
        ty = np.clip(fy - iy, 0.0, 1.0)
        s = self.sdf
        return ((1 - tx) * (1 - ty) * s[ix, iy]
                + tx * (1 - ty) * s[ix + 1, iy]
                + (1 - tx) * ty * s[ix, iy + 1]
                + tx * ty * s[ix + 1, iy + 1])

    # ------------------------------------------------------------------ misc
    def bounds(self, inset: float = 0.0):
        return (np.array([self.xs[0] + inset, self.ys[0] + inset]),
                np.array([self.xs[-1] - inset, self.ys[-1] - inset]))

    def world_to_idx(self, xy):
        p = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        ix = np.rint((p[:, 0] - self.xs[0]) / self.res).astype(int)
        iy = np.rint((p[:, 1] - self.ys[0]) / self.res).astype(int)
        return np.clip(ix, 0, self.nx - 1), np.clip(iy, 0, self.ny - 1)

    def idx_to_world(self, ix, iy):
        return np.stack([self.xs[np.asarray(ix)], self.ys[np.asarray(iy)]], axis=-1)
