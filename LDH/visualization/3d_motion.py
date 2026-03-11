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
# Non Radial Oscillations 3d motion visualization
####################################################################################

input_filename  = ""
output_filename = ".mp4"

l = 
m = 

####################################################################################

data  = np.genfromtxt(input_filename, skip_header=1)
const = np.genfromtxt(input_filename, max_rows   =1)

Rs    = const[1]
omega = const[2]
period = 2 * np.pi / omega

t = np.linspace(0, 2*period, 100)

center_c = [] # in cartesian coordinate
center_s = [] # in spherical coordinate

r     = [1.0]
phi   = np.linspace(0.01, 2*np.pi-0.01, 40)
theta = np.linspace(0.01,   np.pi-0.01, 20)

for ii in range(len(r)) :
    for jj in range(len(theta)) :
        for kk in range(len(phi)) :
            center_s.append( (r[ii], theta[jj], phi[kk]) )
            center_c.append( (r[ii] * np.sin(theta[jj]) * np.cos(phi[kk]),
                              r[ii] * np.sin(theta[jj]) * np.sin(phi[kk]),
                              r[ii] * np.cos(theta[jj])) )

center_s = np.array(center_s)
center_c = np.array(center_c)

N, T = len(center_s), len(t)

scaling = 0.05  ########## 점들의 움직임 진폭 조절

xi_r = [ scaling * data[-1][1]             ]
xi_h = [ scaling * data[-1][2]/Rs/omega**2 ]

dr     = np.zeros((N, T))
dtheta = np.zeros((N, T))
dphi   = np.zeros((N, T))

color_data = np.empty((N, T), dtype=object)

r = np.array(r)
r_to_k = {float(rv): k for k, rv in enumerate(r)}

for ii in range(N) : # number of particles
    kk = r_to_k[float(center_s[ii][0])]
    
    for jj in range(len(t)) : # time
        w = np.cos(omega * t[jj])

        dr[ii][jj]     = np.sqrt( 4 * np.pi ) * xi_r[kk] * sph_harm_y(l, m, center_s[ii][1], center_s[ii][2]).real * w.real
        dtheta[ii][jj] = np.sqrt( 4 * np.pi ) * xi_h[kk] * dYdtheta(  l, m, center_s[ii][1], center_s[ii][2])      * w.real
        dphi[ii][jj]   = np.sqrt( 4 * np.pi ) * xi_h[kk] * dYdphi(    l, m, center_s[ii][1], center_s[ii][2])      * w.real / np.sin( center_s[ii][1] )

        if ( dr[ii][jj] >= 0 ) :
            color_data[ii][jj] = ( "red",   dr[ii][jj]/( np.sqrt( 4 * np.pi ) * xi_r[-1] ) )
        else :
            color_data[ii][jj] = ( "blue", -dr[ii][jj]/( np.sqrt( 4 * np.pi ) * xi_r[-1] ) )

color_data = np.array( color_data )

dx = np.zeros((N, T))
dy = np.zeros((N, T))
dz = np.zeros((N, T))

for ii in range(N) : # number of particles
    for jj in range(len(t)) : # time
        dx[ii][jj]  =     dr[ii][jj] * np.sin(center_s[ii][1]) * np.cos(center_s[ii][2])
        dx[ii][jj] -=   dphi[ii][jj] * np.sin(center_s[ii][2])
        dx[ii][jj] += dtheta[ii][jj] * np.cos(center_s[ii][1]) * np.cos(center_s[ii][2])
        
        dy[ii][jj]  =     dr[ii][jj] * np.sin(center_s[ii][1]) * np.sin(center_s[ii][2])
        dy[ii][jj] +=   dphi[ii][jj] * np.cos(center_s[ii][2])
        dy[ii][jj] += dtheta[ii][jj] * np.cos(center_s[ii][1]) * np.sin(center_s[ii][2])
        
        dz[ii][jj]  =     dr[ii][jj] * np.cos(center_s[ii][1])
        dz[ii][jj] -= dtheta[ii][jj] * np.sin(center_s[ii][1])


animate_from_lists_3d(center_c, t, dx, dy, dz, 
                      colors_t=color_data,
                      outfile=output_filename,
                      fps=25 ########## fps 조절
                     )