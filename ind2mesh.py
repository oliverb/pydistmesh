""" 
Wrapper around distmesh2d and scikit-fmm that generates triangular
meshes from indicator functions.
"""
import numpy
from scipy import weave

import skfmm

from distmesh2d import *

from pylab import *

def ind2mesh(indicator, h0, bbox, pfix=None, maxt=300, maxit=3000, imax=2048, jmax=2048):
    border = 5.0*h0
    xlength, ylength = bbox[1,0]-bbox[0,0]+2*border, bbox[1,1]-bbox[0,1]+2*border

    dx = xlength/imax
    dy = ylength/jmax

# Wrap indicator function to allow some space arund the bounding box
# and transform to 0,0 lower left corner
    def ind_wrap(a, b):
        return indicator(a-bbox[0,0]-border, b-bbox[0,1]-border)

# Fill mesh with indicator values
    ind_mesh = numpy.zeros((imax, jmax))
    for i, x in enumerate(numpy.linspace(0.0, dx*imax, imax)):
        for j, y in enumerate(numpy.linspace(0.0, dy*jmax, jmax)):
            ind_mesh[i,j] = ind_wrap(x,y)

# Compute distance values via fast marching
    phi_mesh = skfmm.distance(ind_mesh, dx=[dx, dy])

# Wrap bilinear interpolation of phi values
    def phi(p):
        return bil_interpolate(p, phi_mesh, dx, dy)

# Generate mesh via distmesh2d
    shift_pfix = pfix + border
    shift_bbox = bbox + border
    p, t = distmesh2d(phi, h0, shift_bbox, pfix=shift_pfix, maxt=maxt, maxit=maxit)

    return p-border, t

def bil_interpolate(p, ls, dx, dy):
    """ 
    2D Bilinear interpolation of values on a uniformly spaced grid
    with mesh width dx and dy in x- and y-direction respectively.
    p can be an array of points as required by distmesh2d.
    """
    m = p.shape[0]
    max_i = ls.shape[0]
    max_j = ls.shape[1]

    z = np.zeros(p.shape[0])
    code = \
    """
    #line 38 "distmesh2d.py"
    int i=0, j=0;
    double x=0.0, y=0.0;

    // Loop  through  all coordinates
    for(int k=0; k<m; k++) {
        // Get grid indices
        i = int( P2(k,0)/dx-0.5 );
        j = int( P2(k,1)/dy-0.5 );

        // Probably not helpful.
        // Just take care that (x,y) lie inside the grid ...
        if(i<0 || i>=max_i || j<0 || j>=max_j) {
            Z1(k) = 100000.0;
            continue;
        }

        // Handle corner case when coordinates lie on border
        if(i==max_i-1) i-=1;
        if(j==max_j-1) j-=1;

        // Local coordinates scaled to [0,1]
        // Maybe this would rather be fmod()?
        x = (P2(k,0)-i*dx)/dx;
        y = (P2(k,1)-j*dy)/dy;

        // Interpolate ls values
        double tmp;
        Z1(k) = LS2(i,j)*(1.0-x)*(1.0-y)
               +LS2(i+1,j)*x*(1.0-y)
               +LS2(i,j+1)*(1-x)*y
               +LS2(i+1,j+1)*x*y;
    }
    """
    weave.inline(code, 
                 ['p', 'ls', 'dx', 'dy', 'z', 'm', 'max_i', 'max_j']) 
    return z

def main():
    from scipy import interpolate

    """Makes indicator function from coefficients c"""
    c = numpy.array([0.15, 0.15])

    n = c.shape[0]
    x = numpy.linspace(1.0, 2.0, n+2)
    y = numpy.zeros(n+2)
    y[1:n+1] = c
    I = interpolate.interp1d(x, y, kind='cubic')

    def chi(x, y):
        # Inside region
        if 0.0 <= x <= 12.0 and 0.0 <= y <= 2.0:
            if x <= 1.0 or x >= 2.0:
                return -1
            ymod = y - 1.0
            Sval = I(x)
            # Could be inside obstacle
            if abs(ymod) <= Sval:
                return 1
            return -1
        return 1

    bounds = numpy.array([[0.0,0.0],[12.0,2.0]])
    pfix = numpy.array([[0.0,0.0],[0.0,2.0],[12.0,0.0],
                        [12.0,2.0],[1.0,1.0],[2.0,1.0]])

    p, t = ind2mesh(chi, 0.1, bounds, pfix)
    export_vtk(p, t, "ind2mesh.vtk")

if __name__=='__main__':
    main()



