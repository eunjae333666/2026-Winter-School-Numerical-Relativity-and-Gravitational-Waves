
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from dataclasses import replace

#constants
dr = 1.0e-3
c = 1.0
m = 1.0
GRID_POINT = 1001
@dataclass
class Parameters :
    n : float = 1.0
    K : float = 1.0e2
    Rho_c : float = 1.28e-3

p = Parameters()

def boun