"""Python port of distmesh2d by Per-Olof Persson and Gilbert Strang"""
import numpy as np
from scipy import spatial
from scipy import sparse

############################################################################# 
###                                                                       ###
### Quasi random halton sequence                                          ###
###                                                                       ###
############################################################################# 

_primes = np.array([
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
       127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
       179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
       233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
       283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
       353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
       419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
       467, 479, 487, 491, 499, 503, 509, 521, 523, 541
        ])

def _halton_next_element(x, eps=10E-12):
    dim, = x.shape
    next_x = np.array([0.0]*dim)

    for i in range(0, dim):
        z = 1.0 - x[i]
        v = 1.0/_primes[i]

        while z < v+eps:
            v = v/_primes[i]

        next_x[i] = x[i] + (_primes[i]+1.0)*v - 1.0

    return next_x

def _halton_seq(N, xl=[0.0, 1.0], yl=[0.0, 1.0]):
    p = np.zeros((N, 2))
    for i in range(0, 1000):
        p[0,:] = _halton_next_element(p[0,:])
    for i in range(1, N):
        p[i,:] = _halton_next_element(p[i-1,:])
    p[:,0] = (xl[1]-xl[0])*p[:,0]+xl[0]
    p[:,1] = (yl[1]-yl[0])*p[:,1]+yl[0]
    return p

############################################################################# 
###                                                                       ###
### Utility functions for distmesh implementation                         ###
###                                                                       ###
############################################################################# 

def unique_rows(a):
    """ 
    Keeps only unique rows in numpy array.
    Found here:
    http://stackoverflow.com/questions/8560440/ ...
    removing-duplicate-columns-and-rows-from-a-numpy-2d-array
    """
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def unit_fh(x):         
    """
    Mesh function corresponding to a uniform mesh.
    """
    return np.ones(x.shape[0])

def _uniform_init(bbox, h0):
    x, y = np.meshgrid(np.arange(bbox[0,0],bbox[1,0],h0),
                       np.arange(bbox[0,1],bbox[1,1],h0))

    x[1::2,:] = x[1::2,:] + h0/2.0 # Shift even rows

    return np.c_[x.flatten(1), y.flatten(1)]

def _halton_init(bbox, h0):
# N is the approx. amount of points a uniform distribution with mesh width h0
# would have.
    N = int((bbox[1,0]-bbox[0,0])/h0*(bbox[1,1]-bbox[0,1])/h0)
    return _halton_seq(N, bbox[:,0], bbox[:,1])                    

initial_values = {"halton"  : _halton_init,
                  "uniform" : _uniform_init}

############################################################################# 
###                                                                       ###
### Distmesh algorithm                                                    ###
###                                                                       ###
############################################################################# 

def distmesh2d(fd, h0, bbox, pfix=None, fh=unit_fh, maxt=None, maxit=None, init="halton"):
    dptol = 0.001
    ttol = .1
    Fscale = 1.2
    deltat = .2
    geps = 0.01*h0
    deps = np.sqrt(0.000001)*h0
    deps = np.sqrt(np.finfo(np.float64).resolution)*h0

### Create initial point distribution
    p = initial_values[init](bbox, h0)

### Remove points outside geometry
    p = p[fd(p) < geps,:]

### Apply rejection method to influence node density
    r0 = 1./fh(p)**2 # Probability to keep point
    r0 = r0/np.max(r0)
    rnd = np.random.uniform(size=p.shape[0])
    p = p[rnd < r0,:]

### Add fixed points
    if pfix is not None:
        p = np.r_[pfix, p]

### TODO
### Duplicate nodes should be dealt with ...

    N = p.shape[0]

### pold will be used to test if its is time for a new triangulation
    pold = p.copy()
    pold[:] = np.inf
    
    triangulations = 0
    iterations = 0

    while True:
### Retriangulation via Delaunay
# Check for "large" movement
        if np.max(np.sqrt(np.sum((p-pold)**2,1))/h0) > ttol:
            pold = p

            tri = spatial.Delaunay(p)
            t = tri.vertices

            pmid = (p[t[:,0],:]+p[t[:,1]]+p[t[:,2]])/3.0 # Compute midpoints
            t = t[fd(pmid) < -geps,:] # Keep interior triangles

# Describe each bar by a unique pair of nodes
# -- To avoid calling extra functions replicating matlab syntax t(:,[2,3]) I
# -- used t[:,0:3:2], which gives the right edges but in the opposite direction!
            bars = np.vstack([t[:,0:2], t[:,1:3], t[:,0:3:2]])
            bars = np.sort(bars,1)
            bars = unique_rows(bars)

            print "Triangulation %d ..." % (triangulations, )
            triangulations += 1
            if maxt is not None:
                if triangulations >= maxt:
                    break

        
### Move mesh points based on bar lenghts L and forces F
        barvec = p[bars[:,0],:]-p[bars[:,1],:] # Bar vectors
        L = np.sqrt(np.sum(barvec**2,1)) # Bar lengths
        barmid = (p[bars[:,0]]+p[bars[:,1],:])/2.0
        hbars = fh(barmid)
        L0 = hbars*Fscale*np.sqrt(np.sum(L**2)/np.sum(hbars**2))

        F = np.maximum(L0-L, np.zeros(L.shape[0]))
        
# Does the same as the sparse.coo_matrix construction below
        #for i in range(0, bars.shape[0]):
        #    # Add force in both directions
        #    Ftot[bars[i,0],:] += deltat*F[i]*barvec[i,:]
        #    Ftot[bars[i,1],:] -= deltat*F[i]*barvec[i,:]

# Workaround ... original: F./L*[1,1].*barvec ...
        Fvec = barvec.copy()
        Fvec[:,0] = F/L*Fvec[:,0]
        Fvec[:,1] = F/L*Fvec[:,1]

        J = np.zeros((F.shape[0],4), dtype=np.int32)
        J[:,1] = 1
        J[:,3] = 1

        I = np.zeros((bars.shape[0],4))
        I[:,0] = bars[:,0]
        I[:,1] = bars[:,0]
        I[:,2] = bars[:,1]
        I[:,3] = bars[:,1]

        As = np.zeros((Fvec.shape[0],4))
        As[:,0] = Fvec[:,0]
        As[:,1] = Fvec[:,1]
        As[:,2] = -Fvec[:,0]
        As[:,3] = -Fvec[:,1]

        Ftot = sparse.coo_matrix((As.flatten(1), (I.flatten(1),J.flatten(1))), shape=(N,2))
        Ftot = np.asarray(Ftot.todense())


# Keep fixed points
        if pfix is not None:
            Ftot[0:pfix.shape[0],:] = 0.0

        p = p + deltat*Ftot

### Bring outside points back to boundary
# This is done via the projection
#   p = p - \phi(p)*\grad \phi(p)
        d = fd(p)
        idx = d>0

        X = p.copy()
        X[idx,0] = X[idx,0] + deps
        dgradx = (fd(X[idx,:])-d[idx])/deps

        Y = p.copy()
        Y[idx,1] = Y[idx,1] + deps
        dgrady = (fd(Y[idx,:])-d[idx])/deps

        X[idx,0] = d[idx]*dgradx
        X[idx,1] = d[idx]*dgrady

        p[idx,:] = p[idx,:] - X[idx,:]


### Check for termination
        if np.max(np.sqrt(np.sum(deltat*Ftot[d<-geps,:]**2,1))) < dptol*h0:
            break
        iterations += 1
        if maxit is not None:
            if iterations >= maxit:
                break

    return p, t

############################################################################# 
###                                                                       ###
### Functions for mesh export                                             ###
###                                                                       ###
############################################################################# 
def _element_quality(p, t):
    """
    Computes element quality of a triangle as 
        Quality(T) = Vol(T)/l_max^2
    where l_max is the longest edge of T.
    """
# Triangle volume
    tvols = 0.5*np.abs( (p[t[:,0],0]-p[t[:,2],0])*(p[t[:,1],1]-p[t[:,0],1])
                          -(p[t[:,0],0]-p[t[:,1],0])*(p[t[:,2],1]-p[t[:,0],1]) )
# Norm along second axis
    def norm(x):
        return np.sqrt(np.sum(x*x, 1))
# Maximal sidelength
    tsides = np.maximum(
                np.maximum( norm(p[t[:,1],:]-p[t[:,0],:]),
                            norm(p[t[:,2],:]-p[t[:,1],:]) ),
                norm(p[t[:,0],:]-p[t[:,2],:]) )

    return tvols/tsides**2

def export_vtk(p, t, filename):
    """
    Exports mesh in .vtk fileformat for viewing in Paraview with element
    quality as cell data.
    """
    import pyvtk

    qual = _element_quality(p, t)

# Add z coordinate of 0.0 to supress vtk warnings
    ps = np.c_[p[:,0], p[:,1], np.zeros(p.shape[0])]

    vtkele = pyvtk.VtkData(
                pyvtk.UnstructuredGrid(
                    ps.tolist(),
                    triangle=t.tolist()),
                pyvtk.CellData(pyvtk.Scalars(qual, lookup_table="default", name="Color")),
                "Mesh")
    vtkele.tofile(filename)

def export_pgf(p, t, filename):
    """ Exports triangulation and element quality in the form
    <
    x_11 y_11 q_1
    x_12 y_12 q_1
    x_13 y_13 q_1

    x_21 y_21 q_2
    x_22 y_22 q_2
    x_23 y_23 q_2

    ...
    >
    Where {x|y}_ij is the {x|y}-coordinate of the j'th vertex if the i'th
    triangle and q_i is its element quality as described in _element_quality.

    This format is suitable for use with pgfplots patchplots.
    """
    qual = _element_quality(p, t)

    with open(filename, "w") as F:
        F.write("x y c\n")
        for i in range(0, t.shape[0]):
            for j in range(0, 3):
                F.write("%f %f %f\n" % (p[t[i,j],0], p[t[i,j],1], qual[i], ))
            F.write("\n")

def export_xml(v, t, filename):
    """ 
    Exports the mesh as xml file suitabele for use with the FEniCS FEM library.
    """
    with open(filename, "w") as tf:
        num_vertices = v.shape[0]
        num_triangles = t.shape[0]
        

        tf.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
        tf.write('<dolfin xmlns:dolfin="http://www.fenics.org/dolfin/">\n')
        tf.write('\t<mesh celltype="triangle" dim="2">\n')
        tf.write('\t\t<vertices size="%d">' % num_vertices)

        for i in range(0, num_vertices):
            tf.write('\t\t\t<vertex index="%d" x="%f" y="%f"/>\n' 
                        % (i, v[i,0], v[i,1]))

        tf.write('\t\t</vertices>\n\t\t<cells size="%d">\n' % num_triangles)

        for i in range(0, num_triangles):
            tf.write('\t\t\t<triangle index="%d" v0="%d" v1="%d" v2="%d"/>\n'
                % (i, t[i,0], t[i,1], t[i,2]))

        tf.write('\t\t</cells>\n\t</mesh>\n</dolfin>')

# 'Testcase'
def _t_phi(x):
    """
    Signed distance function the unit circle.
    """
    return np.maximum(np.sqrt(np.sum(x*x, 1))-1.0, -np.sqrt(np.sum(x*x, 1))+0.5)

_t_bbox = np.array([[-1., -1.],[1., 1.]])

if __name__ == '__main__':
    import pylab as py
    p, t = distmesh2d(_t_phi, 0.05, _t_bbox) 

    export_xml(p, t, "unit_circle.xml")
    export_vtk(p, t, "unit_circle.vtk")
    export_pgf(p, t, "unit_circle.patches")

    py.triplot(p[:,0], p[:,1], t)
    py.show()

                         
