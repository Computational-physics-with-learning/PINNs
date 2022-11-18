import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import os
from tqdm.auto import tqdm


def makedir(p):
    '''
    If p is a file path or directory then this function ensures that the relevant directory exists.
    '''
    d = os.path.dirname(p)
    d = d if len(d) > 0 else p
    os.makedirs(d, exist_ok=True)


def printv(*args, verbosity=np.inf, importance=0, **kwargs):
    '''
    Alias of print except only prints statement if verbosity >= importance
    '''
    if verbosity >= importance:
        print(*args, **kwargs)


def isNull(x):
    if x is None:
        return True
    elif hasattr(x, 'shape'):  # numpy test bluff
        return (x == 0).all()
    else:
        return x == 0


def func1D(A=0, k=0, phi=0, B=0, p=0, psi=0):
    '''
    u = func1D(A=0, k=0, phi=0, B=0, p=0, psi=0)
    Returns function u, a sum of 1-periodic and polynomial terms.

    Parameters:
    -----------
        A=0: float or 1D array of length N
        k=0: float or 1D array of length N
        phi=0: float or 1D array of length N
        B=0: float or 1D array of length M
        p=0: float or 1D array of length M
        psi=0: float or 1D array of length M

    Returns the function u where
        u(x) = sum_{i=0}^{N-1} A[i] * sin(k[i]*x + phi[i]) 
             + sum_{j=0}^{M-1} B[j] * (x - psi[j])**p[j]
    '''
    pms = (A, k, phi, B, p, psi)
    for arr in pms:
        if hasattr(arr, 'shape'):
            arr.shape = (1, -1)

    def f(x):
        x = np.array(x, copy=False).reshape(-1, 1)
        out = 0 * x[:, 0] if isNull(pms[0]) else (pms[0] * np.sin(pms[1] * x + pms[2])).sum(1)
        out += 0 if isNull(pms[3]) else (pms[3] * (x - pms[5])**pms[4]).sum(1)
        return out
    return f


def randParams1D(n, m):
    '''
    Random initialisation of parameters for 'func1D' with n periodic and m polynomial terms.
    '''
    return dict(A=np.random.rand(n) - .5, k=2 * np.pi * (1 + np.random.choice(10, n, False)),
                phi=2 * np.pi * np.random.rand(n), B=np.random.rand(m) - .5,
                p=np.random.choice(10, m, False), psi=np.random.rand(m))


def getEvalPoints2D(Nboundary, Nint):
    '''
    boundary, interior = getEvalPoints2D(Nboundary, Nint)
    Utility for getting a uniform grid of points on the boundary and interior of the domain [0,1]^2

    parameters:
    -----------
        Nboundary: int
            Number of points to place on each boundary
        Nint: int
            Total number of points in interior is Nint^2
    Outputs:
    --------
        boundary : array of shape (4*Nboundary, 2), dtype=float32
            Uniformly sampled points on boundary of [0,1]^2
        interior : array of shape (Nint^2, 2), dtype=float32
            Uniformly sampled points on interior of [0,1]^2
    '''
    tmp = np.linspace(0, 1, Nboundary, endpoint=False)
    boundary = np.empty((4 * Nboundary, 2), dtype='float32')  # points (x,t)
    S = slice(0 * Nboundary, 1 * Nboundary); boundary[S, 0], boundary[S, 1] = tmp, 0  # t=0
    S = slice(1 * Nboundary, 2 * Nboundary); boundary[S, 0], boundary[S, 1] = tmp, 1  # t=1
    S = slice(2 * Nboundary, 3 * Nboundary); boundary[S, 0], boundary[S, 1] = 0, tmp  # x=1
    S = slice(3 * Nboundary, 4 * Nboundary); boundary[S, 0], boundary[S, 1] = 1, tmp  # x=1
    interior = np.linspace(0, 1, Nint + 2)[1:-1].astype('float32')
    interior = np.concatenate(np.meshgrid(interior, interior, [0])[:2], axis=-1).reshape(-1, 2)  # points (x,t)
    return boundary, interior


def splitPoints(X):
    '''
    boundary, interior = splitPoints(X)
    If X is an array of 2D points in [m0,M0]x[m1,M1], then this splits them into those on the boundary and
    those in the interior.
    '''
    m, M = X.min(0), X.max(0)
    boundary = [np.logical_or(X[:, i] == m[i], X[:, i] == M[i]) for i in range(2)]
    boundary = np.logical_or(*boundary)
    return boundary, np.logical_not(boundary)


def params2file(problem, Nsamples, Nboundary, Ninterior, Ncurves):
    if problem.startswith('heat'):
        root = 'heatfuncs'
    elif problem.startswith('advection'):
        root = 'advectionfuncs_periodic' if 'periodic' in problem else 'advectionfuncs'
    stem = f'_{Nsamples}_{Nboundary}_{Ninterior}_{Ncurves}.npz'
    return os.path.join('data', root + stem)


def heatfunc1D(beta=1, A=0, k=0, phi=0, **kwargs):
    '''
    u = heatfunc1D(beta=1, A=0, k=0, phi=0, **kwargs)
    The function u represents a solution to the homogeneous heat equation on [0,1]^2, namely
        \partial_t u = \beta \partial_{xx} u

    Parameters:
    -----------
        A=0: float or 1D array of length N
        k=0: float or 1D array of length N
        phi=0: float or 1D array of length N

    Returns the function u where
        u(t,x) = sum_{i=0}^{N-1} A[i] * sin(k[i]*x + phi[i])exp(-k[i]^2*beta*t)
    More specifically, u(t,x).shape = (t.size, x.size)
    If x is not provided, then it is assumed (t,x) = (t[:,0],t[:,1])
    '''
    pms = (A, k, phi)
    for arr in pms:
        if hasattr(arr, 'shape'):
            arr.shape = (1, 1, -1)

    def u(t, x=None):
        if x is None:
            t, x = t[:, 0].reshape(-1, 1, 1), t[:, 1].reshape(-1, 1, 1)
        else:
            t = np.array(t, copy=False).reshape(-1, 1, 1)
            x = np.array(x, copy=False).reshape(1, -1, 1)
        if isNull(pms[0]):
            return np.zeros((t.size, x.size))
        else:
            return (pms[0] * np.sin(pms[1] * x + pms[2]) * np.exp(-pms[1]**2 * beta * t)).sum(-1)
    return u


def advectionfunc1D(beta=1, **kwargs):
    '''
    u = advectionfunc1D(beta=1, A=0, k=0, phi=0, B=0, p=0, psi=0)
    The function u represents a solution to the homogeneous advection equation on [0,1]^2, namely
        \partial_t u + \beta \partial_x u = 0

    Parameters:
    -----------
        A=0: float or 1D array of length N
        k=0: float or 1D array of length N
        phi=0: float or 1D array of length N

    Returns the function u where
        u(t,x) = u_0(x-t*beta), u_0 = func1D(A,k,phi,B,p,psi)
    More specifically, u(t,x).shape = (t.size, x.size)
    If x is not provided, then it is assumed (t,x) = (t[:,0],t[:,1]), which was the XDE default...
    '''
    f = func1D(**kwargs)

    def u(t, x=None):
        if x is None:
            shape = t.shape[0], 1
            t, x = t[:, 0], t[:, 1]
        else:
            shape = t.size[0], x.size
            t = np.array(t, copy=False).reshape(-1, 1)
            x = np.array(x, copy=False).reshape(1, -1)
        return f(x - t * beta).reshape(shape)
    return u


def plot2d(ax, xy, c, grid=100):
    '''
    Interpolates the values c evaluated at points xy into a regular grid to be plotted on axis ax
    Note that it is assumed c is scaled, imshow will use vmin=0, vmax=1.

    Parameters:
    -----------
        ax: matplotlib object (axis, figure, pyplot) with method ax.imshow
        xy: array of shape (N,2), points corresponding to values c
        c: array of shape (N,), values corresponding to points xy
        grid: int, default 100. Image will be plotted on uniform grid of size gridxgrid
    '''
    X, Y = np.meshgrid(*[np.linspace(0, 1, grid, endpoint=False)] * 2)
    Z = griddata(xy, c, (X, Y), method='linear', )
    return ax.imshow(Z.T, origin='lower', extent=[0, 1, 0, 1], vmin=0, vmax=1)


def plotcompare2d(TX, U, V, fig=None):
    '''
    Interpolates the values c evaluated at points xy into a regular grid to be plotted on axis ax
    Note that it is assumed c is scaled, imshow will use vmin=0, vmax=1.

    Parameters:
    -----------
        ax: matplotlib object (axis, figure, pyplot) with method ax.imshow
        xy: array of shape (N,2), points corresponding to values c
        c: array of shape (N,), values corresponding to points xy
        grid: int, default 100. Image will be plotted on uniform grid of size gridxgrid
    '''

    fig = plt.gcf() if fig is None else fig
    rescale = U.min(), U.ptp()
    U, V = ((arr - rescale[0]) / rescale[1] for arr in (U, V))
    ax = fig.subplots(1, 3)
    plot2d(ax[0], TX, U)
    ax[0].set_title('Ground-truth'); ax[0].set_ylabel('time'); ax[0].set_xticks([])
    plot2d(ax[1], TX, V)
    ax[1].set_title('Reconstruction, average error=%.1e' % abs(U - V).mean())
    ax[1].set_xticks([]); ax[1].set_yticks([])
    t0 = (TX[:, 0] == 0)
    ax[2].plot(U[t0], label='true')
    ax[2].plot(V[t0], label='recon')
    ax[2].set_title('Initial condition, average boundary error=%.1e' % abs(U[t0] - V[t0]).mean()
                    ); ax[2].set_xticklabels([]); plt.legend(bbox_to_anchor=(.85, 0.0), ncol=2)
    plt.tight_layout()


def examples2D(which, Nsamples=2e3, Nboundary=1e3, Ninterior=1e2, Ncurves=5, seed=101, **kwargs):
    '''
    Gets examples of solutions to basic PDEs.

    Parameters:
    -----------
        which: str, either 'heat', 'advection-periodic', or 'advection'
            Which pde variant to use
        Nsamples: int
            Number of examples to generate
        Nboundary: int
            Number of points to sample on each boundary
    '''
    if seed is not None:
        np.random.seed(seed)
    N = (Nsamples, Nboundary, Ninterior, Ncurves) = [int(n) for n in (Nsamples, Nboundary, Ninterior, Ncurves)]
    p = params2file(which, *N)

    if not os.path.exists(p):
        x = np.concatenate(getEvalPoints2D(Nboundary, Ninterior), axis=0)

        if which == 'heat':
            beta = kwargs.pop('beta', 0.01)
            func = heatfunc1D
            F = 'A', 'k'
        elif which == 'advection-periodic':
            beta = kwargs.pop('beta', 2.0)
            func = advectionfunc1D
            F = 'A', 'k', 'phi'
        elif which == 'advection':
            beta = kwargs.pop('beta', 1.0)
            func = advectionfunc1D
            F = 'A', 'k', 'phi', 'B', 'p', 'psi'
        kwargs['beta'] = beta

        U = np.empty((Nsamples, x.shape[0]), dtype='float32')
        F = {k: np.empty((Nsamples, Ncurves), dtype='float32') for k in F}

        for i in tqdm(range(Nsamples)):
            f = randParams1D(Ncurves, Ncurves)  # get random parameters
            f = {k: f[k] for k in F}  # remove unwanted ones
            for k in F:
                F[k][i] = f[k]  # store function parameters
            f = func(**kwargs, **f)  # compile function
            U[i] = f(x)[:, 0]  # store function values
        makedir(p)
        np.savez(p, U=U, x=x, **kwargs, **F)

    return np.load(p)
