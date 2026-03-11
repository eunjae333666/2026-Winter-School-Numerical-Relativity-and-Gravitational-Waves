import numpy as np
import ffmpeg

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

from scipy.special import sph_harm_y

from pathlib import Path
from typing import Sequence, Union, Tuple, Optional

try:
    from matplotlib.animation import FFMpegWriter  # type: ignore
    _HAS_FFMPEG = True
except Exception:
    FFMpegWriter = None  # type: ignore
    _HAS_FFMPEG = False

ArrayLike = Union[Sequence[float], np.ndarray]

def _to_ndarray_xy(centers: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(centers, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("centers must have shape (N, 2)")
    return arr


def _expand_time(N: int, t: ArrayLike, *arrays: ArrayLike) -> tuple:
    """Normalize t, and expand every array to shape (N,T).
    Each array may be 1D (T,) or 2D (N,T). Returns (t_sorted, *expanded).
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be 1D")
    T = t.shape[0]

    # sort by time if needed
    if not np.all(np.diff(t) >= 0):
        idx = np.argsort(t)
        t = t[idx]
        arrays = tuple((np.asarray(a)[:, idx] if np.asarray(a).ndim == 2 else np.asarray(a)[idx]) for a in arrays)
    else:
        arrays = tuple(np.asarray(a) for a in arrays)

    def expand(a: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            if a.shape[0] != T:
                raise ValueError("1D time‑series must have length len(t)")
            return np.broadcast_to(a, (N, T))
        if a.ndim == 2:
            if a.shape != (N, T):
                raise ValueError("2D time‑series must have shape (N, T)")
            return a
        raise ValueError("time‑series must be 1D or 2D")

    return (t, *[expand(a) for a in arrays])


def animate_from_lists_2d(
    centers: Sequence[Sequence[float]],
    t: ArrayLike,
    dx: ArrayLike,
    dy: ArrayLike,
    *,
    vx: Optional[ArrayLike] = None,
    vy: Optional[ArrayLike] = None,
    vel_scale: float = 1.0,
    vel_normalize: bool = False,
    outfile: Union[str, Path] = "motion2d.mp4",
    fps: int = 30,
    sizes: Union[float, Sequence[float]] = 36.0,
    title: Optional[str] = None,
    dpi: int = 150,
    show: bool = False,
    transparent: bool = False,
) -> None:
    centers_arr = _to_ndarray_xy(centers)
    N = centers_arr.shape[0]

    # Expand mandatory series
    t_arr, dx_arr, dy_arr = _expand_time(N, t, dx, dy)

    # Expand velocity series if provided; else zeros (no arrows)
    if vx is None: vx = np.zeros_like(dx_arr)
    if vy is None: vy = np.zeros_like(dy_arr)
    (_, vx_arr, vy_arr) = _expand_time(N, t_arr, vx, vy)

    # frames: use data length so video length ~= len(t)/fps seconds
    nframes = int(t_arr.shape[0])

    # Figure/Axes
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

    if transparent:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((0,0,0,0))
        saveface = {"transparent": True}
    else:
        fig.patch.set_alpha(1.0)
        fig.set_facecolor("white")
        ax.set_facecolor("white")
        saveface = {"transparent": False, "facecolor": "white"}

    # Clean look
    ax.grid(False)
    if title:
        ax.set_title(title)

    # Fixed limits from full envelopes
    rmax = float(max(np.max(np.abs(dx_arr)), np.max(np.abs(dy_arr))))
    xmin = float(np.min(centers_arr[:, 0]) - rmax)
    xmax = float(np.max(centers_arr[:, 0]) + rmax)
    ymin = float(np.min(centers_arr[:, 1]) - rmax)
    ymax = float(np.max(centers_arr[:, 1]) + rmax)
    # add small margin
    xspan = xmax - xmin; yspan = ymax - ymin
    xmin -= 0.05 * xspan; xmax += 0.05 * xspan
    ymin -= 0.05 * yspan; ymax += 0.05 * yspan
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')

    # sizes handling for scatter (pt^2)
    if isinstance(sizes, (int, float)):
        sizes_arr = np.full(N, float(sizes))
    else:
        sizes_arr = np.asarray(sizes, dtype=float)
        if sizes_arr.shape[0] != N:
            raise ValueError("sizes length must equal number of particles")

    def update(frame: int):
        # positions
        x = centers_arr[:, 0] + dx_arr[:, frame]
        y = centers_arr[:, 1] + dy_arr[:, frame]
        # velocities
        u = vx_arr[:, frame]
        v = vy_arr[:, frame]

        ax.clear()
        if not transparent:
            ax.set_facecolor("white")
        if title:
            ax.set_title(title)
        ax.grid(False)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal', adjustable='box')

        # points (black)
        ax.scatter(x, y, s=sizes_arr, c=["black"] * N)
        # velocity arrows (black)
        # scale_units='xy' keeps arrow length in data units; scale controls 1/length
        if vel_normalize:
            # normalize: make all arrows same length (use unit vectors)
            mag = np.hypot(u, v)
            uu = np.divide(u, mag, out=np.zeros_like(u), where=mag>0)
            vv = np.divide(v, mag, out=np.zeros_like(v), where=mag>0)
            # ax.quiver(x, y, uu, vv, angles='xy', scale_units='xy', scale=None, color='black')
            ax.quiver(x, y, uu, vv, angles='xy', scale_units='xy', scale=1/float(vel_scale), color='black')
        else:
            # ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=None, color='black')
            ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1/float(vel_scale), color='black')
        return ()

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000.0 / fps, blit=False)

    if show:
        plt.show()
        return

    outfile = Path(outfile)
    if outfile.suffix.lower() == ".mp4" and _HAS_FFMPEG:
        writer = FFMpegWriter(fps=fps, metadata={"artist": "animate_particles_2d_vel"})
        anim.save(str(outfile), writer=writer, dpi=dpi, savefig_kwargs=saveface)
    else:
        if outfile.suffix.lower() != ".gif":
            outfile = outfile.with_suffix(".gif")
        writer = PillowWriter(fps=fps)
        anim.save(str(outfile), writer=writer, dpi=dpi, savefig_kwargs=saveface)

    plt.close(fig)


__all__ = ["animate_from_lists_2d"]



def _to_ndarray_xyz(centers: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(centers, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3)")
    return arr


def _normalize_motion(N: int, t: ArrayLike, dx: ArrayLike, dy: ArrayLike, dz: ArrayLike):
    t = np.asarray(t, dtype=float)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    dz = np.asarray(dz, dtype=float)

    if t.ndim != 1:
        raise ValueError("t must be 1D")
    T = t.shape[0]

    if not np.all(np.diff(t) >= 0):
        idx = np.argsort(t)
        t = t[idx]
        dx = dx[:, idx] if dx.ndim == 2 else dx[idx]
        dy = dy[:, idx] if dy.ndim == 2 else dy[idx]
        dz = dz[:, idx] if dz.ndim == 2 else dz[idx]

    def expand(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            if arr.shape[0] != T:
                raise ValueError("dx/dy/dz 1D must have length len(t)")
            return np.broadcast_to(arr, (N, T))
        if arr.ndim == 2:
            if arr.shape != (N, T):
                raise ValueError("dx/dy/dz 2D must have shape (N, T)")
            return arr
        raise ValueError("dx/dy/dz must be 1D or 2D")

    return t, expand(dx), expand(dy), expand(dz)


def _convert_color_elem(elem, cmap_obj, norm):
    # string only
    if isinstance(elem, str):
        return mcolors.to_rgba(elem)
    # tuple (string, alpha)
    if isinstance(elem, (tuple, list)) and len(elem) == 2 and isinstance(elem[0], str):
        base = mcolors.to_rgba(elem[0])
        return (base[0], base[1], base[2], float(elem[1]))
    # numeric
    try:
        val = float(elem)
        return cmap_obj(norm(val))
    except Exception:
        raise ValueError(f"Unsupported color element: {elem}")


def _normalize_colors(N: int, T: int, colors_t, cmap: str, vmin, vmax):
    if colors_t is None:
        def provider(frame):
            return ["black"] * N
        return provider

    arr = np.asarray(colors_t, dtype=object)
    if arr.ndim == 1:
        if arr.shape[0] != T:
            raise ValueError("colors_t 1D must have length len(t)")
        arr = np.broadcast_to(arr, (N, T))
    elif arr.ndim == 2:
        if arr.shape != (N, T):
            raise ValueError("colors_t 2D must have shape (N, T)")
    else:
        raise ValueError("colors_t must be 1D or 2D")

    # check numeric possibility
    flat = arr.ravel()
    all_num = True
    vals = []
    for e in flat:
        try:
            vals.append(float(e))
        except Exception:
            all_num = False
            break
    if all_num:
        arr_num = np.array(vals).reshape(arr.shape)
        if vmin is None: vmin = float(np.nanmin(arr_num))
        if vmax is None: vmax = float(np.nanmax(arr_num))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        # cmap_obj = cm.get_cmap(cmap)
        cmap_obj = mpl.colormaps[cmap]
        def provider(frame):
            return [cmap_obj(norm(v)) for v in arr_num[:, frame]]
        return provider

    # else treat as mixed string or (string,alpha)
    # cmap_obj = cm.get_cmap(cmap)
    cmap_obj = mpl.colormaps[cmap]
    norm = mcolors.Normalize(vmin=0, vmax=1)
    def provider(frame):
        return [_convert_color_elem(e, cmap_obj, norm) for e in arr[:, frame]]
    return provider


def animate_from_lists_3d(
    centers: Sequence[Sequence[float]],
    t: ArrayLike,
    dx: ArrayLike,
    dy: ArrayLike,
    dz: ArrayLike,
    *,
    colors_t=None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    outfile: Union[str, Path] = "motion3d.mp4",
    fps: int = 30,
    sizes: Union[float, Sequence[float]] = 36.0,
    title: Optional[str] = None,
    dpi: int = 150,
    show: bool = False,
    transparent: bool = False,
) -> None:
    centers_arr = _to_ndarray_xyz(centers)
    N = centers_arr.shape[0]
    t_arr, dx_arr, dy_arr, dz_arr = _normalize_motion(N, t, dx, dy, dz)

    nframes = int(t_arr.shape[0])

    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        saveface = {"transparent": True}
    else:
        fig.patch.set_alpha(1.0)
        fig.set_facecolor("white")
        ax.set_facecolor("white")
        saveface = {"transparent": False, "facecolor": "white"}

    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)
    if title:
        ax.set_title(title)

    rmax = float(max(np.max(np.abs(dx_arr)), np.max(np.abs(dy_arr)), np.max(np.abs(dz_arr))))
    xmin = float(np.min(centers_arr[:, 0]) - rmax)
    xmax = float(np.max(centers_arr[:, 0]) + rmax)
    ymin = float(np.min(centers_arr[:, 1]) - rmax)
    ymax = float(np.max(centers_arr[:, 1]) + rmax)
    zmin = float(np.min(centers_arr[:, 2]) - rmax)
    zmax = float(np.max(centers_arr[:, 2]) + rmax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect([1, 1, 1])

    init_elev, init_azim = ax.elev, ax.azim

    if isinstance(sizes, (int, float)):
        sizes_arr = np.full(N, float(sizes))
    else:
        sizes_arr = np.asarray(sizes, dtype=float)
        if sizes_arr.shape[0] != N:
            raise ValueError("sizes length must equal number of particles")

    color_provider = _normalize_colors(N, t_arr.shape[0], colors_t, cmap, vmin, vmax)

    def update(frame: int):
        x = centers_arr[:, 0] + dx_arr[:, frame]
        y = centers_arr[:, 1] + dy_arr[:, frame]
        z = centers_arr[:, 2] + dz_arr[:, frame]

        ax.clear()
        if not transparent:
            ax.set_facecolor("white")
        if title:
            ax.set_title(title)
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.grid(False)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=init_elev, azim=init_azim)

        c_frame = color_provider(frame)
        ax.scatter(x, y, z, s=sizes_arr, c=c_frame)
        return ()

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000.0 / fps, blit=False)

    if show:
        plt.show()
        return

    outfile = Path(outfile)
    if outfile.suffix.lower() == ".mp4" and _HAS_FFMPEG:
        writer = FFMpegWriter(fps=fps, metadata={"artist": "animate_particles_3d"})
        anim.save(str(outfile), writer=writer, dpi=dpi, savefig_kwargs=saveface)
    else:
        if outfile.suffix.lower() != ".gif":
            outfile = outfile.with_suffix(".gif")
        writer = PillowWriter(fps=fps)
        anim.save(str(outfile), writer=writer, dpi=dpi, savefig_kwargs=saveface)


    plt.close(fig)

__all__ = ["animate_from_lists_3d"]