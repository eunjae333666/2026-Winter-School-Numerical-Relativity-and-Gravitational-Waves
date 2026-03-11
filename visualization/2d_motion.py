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

from animation import animate_from_lists_2d
from animation import animate_from_lists_3d

def dYdtheta ( l, m, theta, phi ) :
    eps = 1e-6
    return ( sph_harm_y(l, m, theta + eps, phi).real - sph_harm_y(l, m, theta - eps, phi).real ) / (2 * eps)

def dYdphi ( l, m, theta, phi ) :
    eps = 1e-6
    return ( sph_harm_y(l, m, theta, phi + eps).real - sph_harm_y(l, m, theta, phi - eps).real ) / (2 * eps)

def interpolation_bkg(x, data_x, data_y) :
    if 1 < x :
        return data_y[-1]

    if x <= 0 :
        return data_y[0]
    
    for ii in range(len(data_x)) :
        if ( data_x[ii] < x and x <= data_x[ii+1] ) :
            break

    x1 = data_x[ii]
    x2 = data_x[ii+1]
    y1 = data_y[ii]
    y2 = data_y[ii+1]
    
    return y1 + (y2-y1)/(x2-x1) * (x-x1)

def interpolation_osc(x, data_x, data_y, L) :
    if 1 < x :
        return data_y[-1]

    if x <= 0 :
        return 0
    
    for ii in range(len(data_x)) :
        if ( data_x[ii] < x and x <= data_x[ii+1] ) :
            break

    x1 = data_x[ii]
    x2 = data_x[ii+1]
    y1 = data_y[ii]   * data_x[ii]   ** L
    y2 = data_y[ii+1] * data_x[ii+1] ** L
    
    return y1 + (y2-y1)/(x2-x1) * (x-x1)

####################################################################################
# Non Radial Oscillations 2d motion & density perturbation visualization
####################################################################################
input_filename_background  = "problem1.dat"   # 문제 1번 결과 파일명 [cite: 792]
input_filename_oscillation = "problem3.dat"   # 문제 3번 결과 파일명 [cite: 794]
output_filename = "problem3_2d_xz.gif"       # 저장될 영상 이름 [cite: 794]

l = 2  # 사중극자 모드 (문제 3-4 힌트) [cite: 41, 795]
m = 2  # 0 <= m <= l 범위 내 정수 (보통 2 사용) [cite: 41, 795]

plane = "xz"  # "xy", "xz", "yz" 중 보고 싶은 단면 [cite: 796

####################################################################################

n = 3.0 ** (1.0/2.0)
K = 3.0

data_bkg = np.genfromtxt(input_filename_background,  skip_header=1 )
data_osc = np.genfromtxt(input_filename_oscillation, skip_header=1 )
const    = np.genfromtxt(input_filename_oscillation, max_rows   =1 )

omega  = const[2]
period = 2*np.pi/omega

T = 100
t_vals = np.linspace(0, 2*period, T)


u = np.linspace(-1.005, 1.005, 202)
U, V = np.meshgrid(u, u, indexing="xy")

if plane == "xy":
    x, y, z = U, V, np.zeros_like(U)
elif plane == "xz":
    x, y, z = U, np.zeros_like(U), V
elif plane == "yz":
    x, y, z = np.zeros_like(U), U, V
else:
    raise ValueError("plane은 'xy', 'xz', 'yz' 중 하나여야 합니다.")

r = np.sqrt(x*x + y*y + z*z)
mask = (r <= 1.0)

x_m = x[mask]; y_m = y[mask]; z_m = z[mask]; r_m = r[mask]
theta_m = np.arccos(z_m / r_m)
phi_m   = np.mod(np.arctan2(y_m, x_m), 2*np.pi)

h  = np.array([interpolation_bkg(rt, data_bkg[:,0], data_bkg[:,1])    for rt in r_m])
dh = np.array([interpolation_osc(rt, data_osc[:,0], data_osc[:,2], l) for rt in r_m])

drho_A_in    = n * h**(n-1) * dh / (K*(n+1))**n
drho_A       = np.zeros_like(r, dtype=float)
drho_A[mask] = drho_A_in

Ylm_real = np.sqrt(4*np.pi) * sph_harm_y(l, m, theta_m, phi_m).real
Ylm_real = np.where(np.abs(Ylm_real) < 1e-15, 0.0, Ylm_real)

z_grid = np.full((T,) + r.shape, np.nan, dtype=float)
for ii in range(T):
    w = np.cos(omega * t_vals[ii])
    z_grid[ii][mask] = drho_A[mask] * Ylm_real * w

vmin = np.nanmin(z_grid[:, mask])
vmax = np.nanmax(z_grid[:, mask])


data_xi = np.genfromtxt(input_filename_oscillation)

dx_grid = 1.0/10.0
dy_grid = 1.0/10.0
x_vals = -1.05 + dx_grid * np.arange(1, 21)
y_vals = -1.05 + dy_grid * np.arange(1, 21)

X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')
R2 = X**2 + Y**2
inside = (R2 <= 1.0)

if plane == "xy":
    r2d     = np.sqrt(R2)
    theta2d = (np.pi/2) * np.ones_like(r2d)
    phi2d   = np.arctan2(Y, X)
elif plane == "xz":
    r2d     = np.sqrt(X**2 + Y**2)
    phi2d   = np.where(X >= 0, 0.0, np.pi)
    theta2d = np.arctan2(np.abs(X), Y)
elif plane == "yz":
    r2d     = np.sqrt(X**2 + Y**2)
    phi2d   = np.where(X >= 0, np.pi/2, 3*np.pi/2)
    theta2d = np.arctan2(np.abs(X), Y)
else:
    raise ValueError("plane에는 'xy', 'yz', 'xz' 세 가지 중 하나를 넣어주세요.")

rr = r2d[inside]
tt = theta2d[inside]
pp = phi2d[inside]

center_c = np.stack([
    rr * np.sin(tt) * np.cos(pp),
    rr * np.sin(tt) * np.sin(pp),
    rr * np.cos(tt)
], axis=1)
N = len(rr)

scaling = 0.02
xi_r_at_R = data_xi[-1][0] * scaling
xi_h_at_R = data_xi[-1][2] * scaling

coswt = np.cos(omega * t_vals)
pref = np.sqrt(4*np.pi)

Yreal = sph_harm_y(l, m, tt, pp).real
dY_th = dYdtheta(  l, m, tt, pp)
dY_ph = dYdphi(    l, m, tt, pp)

dr     = (pref * xi_r_at_R * Yreal)[:, None] * coswt[None, :]
dtheta = (pref * xi_h_at_R * dY_th)[:, None] * coswt[None, :]
dphi   = (pref * xi_h_at_R * dY_ph)[:, None] * coswt[None, :]
dphi   = dphi / np.sin(tt)[:, None]

sin_tt = np.sin(tt)[:, None]
cos_tt = np.cos(tt)[:, None]
sin_pp = np.sin(pp)[:, None]
cos_pp = np.cos(pp)[:, None]

dx = dr * sin_tt * cos_pp - dphi * sin_pp + dtheta * cos_tt * cos_pp
dy = dr * sin_tt * sin_pp + dphi * cos_pp + dtheta * cos_tt * sin_pp
dz = dr * cos_tt          - dtheta * sin_tt

dt = t_vals[1] - t_vals[0]
vx = np.zeros((N, T)); vy = np.zeros((N, T)); vz = np.zeros((N, T))
vx[:, 1:-1] = (dx[:, 2:] - dx[:, :-2])/(2*dt)
vy[:, 1:-1] = (dy[:, 2:] - dy[:, :-2])/(2*dt)
vz[:, 1:-1] = (dz[:, 2:] - dz[:, :-2])/(2*dt)

scaling2 = 0.5   ########## 점들의 움직임 진폭 조절
dx *= scaling2; dy *= scaling2; dz *= scaling2
vx *= scaling2; vy *= scaling2; vz *= scaling2

if plane == "xy":
    base2d = center_c[:, [0, 1]]
    p1 = base2d[:, 0][:, None] + dx
    p2 = base2d[:, 1][:, None] + dy
    v1, v2 = vx, vy
elif plane == "xz":
    base2d = center_c[:, [0, 2]]
    p1 = base2d[:, 0][:, None] + dx
    p2 = base2d[:, 1][:, None] + dz
    v1, v2 = vx, vz
elif plane == "yz":
    base2d = center_c[:, [1, 2]]
    p1 = base2d[:, 0][:, None] + dy
    p2 = base2d[:, 1][:, None] + dz
    v1, v2 = vy, vz


fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

im = ax.imshow(
    z_grid[0],
    origin="lower",
    extent=[u.min(), u.max(), u.min(), u.max()],
    vmin=vmin, vmax=vmax,
    cmap="viridis",
    interpolation="bilinear",
    alpha=1.0
)

ax.add_patch(patches.Circle((0, 0), 1.0, fill=False, linewidth=1.0, color="k"))

sc = ax.scatter(p1[:, 0], p2[:, 0], s=30, c="red", edgecolors="none", zorder=3)

qv = ax.quiver(
    p1[:, 0], p2[:, 0],
    v1[:, 0], v2[:, 0],
    color="red",
    angles="xy", scale_units="xy",
    scale=0.008,   ########## 화살표 크기 조절 - scale이 클수록 짧은  화살표
    width=0.005,    ########## 화살표 넓이 조절 - width가 클수록 두꺼운 화살표
    zorder=10
)

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

cb = fig.colorbar(
    im, ax=ax,
    shrink=0.75,
    fraction=0.045,
    pad=0.03,
    aspect=25,
    label="density perturbation"
)

def update(frame):
    im.set_data(z_grid[frame])

    # scatter 위치 갱신
    sc.set_offsets(np.c_[p1[:, frame], p2[:, frame]])

    # quiver 위치/벡터 갱신
    qv.set_offsets(np.c_[p1[:, frame], p2[:, frame]])
    qv.set_UVC(v1[:, frame], v2[:, frame])

    ax.set_title(f"{plane}-plane | t = {t_vals[frame]:.4f}")
    return (im, sc, qv)

ani = FuncAnimation(fig, update, frames=T, interval=100, blit=False)
ani.save(output_filename, fps=25, dpi=120) ,    ########## fps, dpi 조절
plt.close(fig)