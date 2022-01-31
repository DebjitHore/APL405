"""
Mohr circle in 3D.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh
from matplotlib import rcParams
 
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

class mohr():
  def plot_mohr3d(self, S):
    """Plot 3D Mohr circles."""
    
    S3, S2, S1 = eigvalsh(S) #Eigen values of the stress tensor in increasing order of magnitude i.e. Principal Stresses.

    R_maj = 0.5*(S1 - S3)    #Major radius
    cent_maj = 0.5*(S1+S3)   #Centre of the major circle

    R_min = 0.5*(S2 - S3)    #Minor Radius
    cent_min = 0.5*(S2 + S3) #Centre of the minor circle
    
    R_mid = 0.5*(S1 - S2)   #Intermediate
    cent_mid = 0.5*(S1 + S2)
    
    #FIGURE
    circ1 = plt.Circle((cent_maj,0), R_maj, facecolor='#cce885', lw=3,
                       edgecolor='#5c8037')
    circ2 = plt.Circle((cent_min,0), R_min, facecolor='w', lw=3,
                       edgecolor='#15a1bd')
    circ3 = plt.Circle((cent_mid,0), R_mid, facecolor='w', lw=3,
                       edgecolor='#e4612d')
    plt.axis('image')
    ax = plt.gca()
    ax.add_artist(circ1)
    ax.add_artist(circ2)
    ax.add_artist(circ3)
    ax.set_xlim(S3 - .1*R_maj, S1 + .1*R_maj)
    ax.set_ylim(-1.1*R_maj, 1.1*R_maj)
    plt.xlabel(r"$\sigma$", size=18)
    plt.ylabel(r"$\tau$", size=18)
    #plt.savefig('Mohr_circle_3D.svg')
    plt.show()

    return R_maj, R_min, R_mid


S = np.array([
    [90,0,95],
    [0,96,0],
    [95,0,-50]])    

plt.figure()
model = mohr()
model.plot_mohr3d(S)

S2 = np.array([
    [1,0,0],
    [0,2,0],
    [0,0,4]])  

plt.figure()
model.plot_mohr3d(S2)
plt.show()

