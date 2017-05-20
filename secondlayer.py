import numpy as np

from numba import jit  # playing with JIT compiler


def _gdummyv_vanilla(G, g, m, n, error,memarray):
    """Chris' original version. """
    inp = int(2*m)
    outn = int(n)
    for jj in np.arange(1, inp+2):  # ???: Missing the last value on purpose?
        for fxn in np.arange(1,outn +1):
            Gmax = memarray[jj - 1,fxn - 1,0]
            Gmin = memarray[jj - 1,fxn - 1,1]
            ndevrange = Gmax- Gmin
            dg = ndevrange/(g-1)
            if error[jj - 1][fxn - 1] > 0:
                G[jj - 1][fxn - 1] = G[jj - 1][fxn - 1] + dg
            elif error[jj - 1][fxn - 1] < 0:
                G[jj - 1][fxn - 1] = G[jj - 1][fxn - 1] - dg
             
            if G[jj - 1][fxn - 1] >  Gmax:
               G[jj - 1][fxn - 1] = Gmax
            elif G[jj - 1][fxn - 1] <  Gmin:
                G[jj - 1][fxn - 1] = Gmin

    return G


def _gdummyv_numpy(G, g, m, n, error,memarray):
    """Beware: this implementation does not really bother about the
    temporary copies of possibly heavy arrays...
    """
    inp = int(2 * m)
    outn = int(n)

    # Beware that we first have to have to extract subviews of the
    # relevant arrays as we avoid the last row of G (but why do we?!)
    G_view = G[0:inp + 1, 0:outn]
    error_view = error[0:inp + 1, 0:outn]
    Gmax_view = memarray[0:inp + 1, 0:outn, 0]
    Gmin_view = memarray[0:inp + 1, 0:outn, 1]

    dg = (Gmax_view - Gmin_view) / (g - 1.0)
    selection = error_view > 0
    G_view[selection] += dg[selection]
    selection = error_view < 0
    G_view[selection] -= dg[selection]

    # NB: in-place operation to directly modify G.
    np.clip(G_view, Gmin_view, Gmax_view, out=G_view)

    return G


@jit(nopython=True)
def _gdummyv_jit(G, g, m, n, error, memarray):

    inp = int(2 * m)
    outn = int(n)

    for jj in range(inp + 1):
        for fxn in range(outn):

            Gmax = memarray[jj, fxn, 0]
            Gmin = memarray[jj, fxn, 1]
            dg = (Gmax - Gmin) / (g - 1.0)

            if error[jj, fxn] > 0:
                G[jj, fxn] += dg
            elif error[jj, fxn] < 0:
                G[jj, fxn] -= dg

            # Tip: if one were not using numba but only numpy
            # one could just call use something like:
            # ``return np.clip(G, Gmin, Gmax, out=G)``
            if G[jj, fxn] >  Gmax:
                G[jj, fxn] = Gmax
                continue
            if G[jj, fxn] <  Gmin:
                G[jj, fxn] = Gmin

    return G


def gdummyv(G, g, m, n, error, memarray):
    """TODO: write a docstring!

    Parameters
    ----------
    G : ndarray
        Conductance values of the synaptic crossbar.
        Its shape is (2*m + 2, n).

    g : int
        Number of writable synaptic levels.

    m, n : int
        Number of features, resp. classes.

    error : ndarray
        ??? DESCRIPTION!

    memarray : ndarray
        ??? DESCRIPTION!

    Returns
    -------
    up-to-date G : ndarray
    """

    return _gdummyv_jit(G, g, m, n, error, memarray)


def _gdummynv_vanilla(G, gmax, gmin, g, m, n, error):
    """Chris' original version. """
    ndevrange = gmax- gmin
    dg = ndevrange/(g-1)
    inp = int(2*m)
    outn = int(n)
    for jj in np.arange(1,inp+2):  # ???: Missing the last value on purpose?
        for fxn in np.arange(1,outn+1):
            if error[jj - 1][fxn - 1] > 0:
                G[jj - 1][fxn - 1] = G[jj - 1][fxn - 1] + dg
            elif error[jj - 1][fxn - 1] < 0:
                G[jj - 1][fxn - 1] = G[jj - 1][fxn - 1] - dg
                
    G[G < gmin] = gmin
    G[G > gmax] = gmax
    return G


def _gdummynv_numpy(G, gmax, gmin, g, m, n, error):
    """Beware: this implementation does not really bother about the
    temporary copies of possibly heavy arrays...
    """
    inp = int(2 * m)
    outn = int(n)
    shape = (inp + 1, outn)

    dg = (gmax - gmin) / (g - 1.0)

    # Beware that we first have to have to extract subviews of the
    # relevant arrays as we avoid the last row of G (but why do we?!)
    G_view = G[0:inp + 1, 0:outn]
    error_view = error[0:inp + 1, 0:outn]

    selection = error_view > 0
    G_view[selection] += dg
    selection = error_view < 0
    G_view[selection] -= dg

    G = np.clip(G, gmin, gmax)

    return G

@jit(nopython=True)
def _gdummynv_jit(G, gmax, gmin, g, m, n, error):

    inp = int(2 * m)
    outn = int(n)

    dg = (gmax - gmin) / (g - 1.0)

    for jj in range(inp + 1):
        for fxn in range(outn):

            if error[jj, fxn] > 0:
                G[jj, fxn] += dg
                continue
            if error[jj, fxn] < 0:
                G[jj, fxn] -= dg

    # One more time, if using only numpy, one may want to simply use:
    # ``return np.clip(G, Gmin, Gmax, out=G)``
    for row in range(G.shape[0]):
        for col in range(G.shape[1]):
            if G[row, col] < gmin:
                G[row, col] = gmin
                continue  # skip the next test for speed
            if G[row, col] > gmax:
                G[row, col] = gmax

    return G


def gdummynv(G, gmax, gmin, g, m, n, error):
    """TODO: write a docstring!

    Parameters
    ----------
    G : ndarray
        Conductance values of the synaptic crossbar.
        Its shape is (2*m + 2, n).

    gmax, gmin : floats
        Maximal, resp. minimal, value of conductance.

    g : int
        Number of writable synaptic levels.

    m, n : int
        Number of features, resp. classes.

    error : ndarray
        DESCRIPTION!

    Returns
    -------
    up-to-date G : ndarray
    """
    return _gdummynv_jit(G, gmax, gmin, g, m, n, error)


def _gdummynvana_vanilla(G, gmax, gmin, g, m, n, error,deltas):
    """Chris' original version. """
    ndevrange = gmax- gmin
    dg = ndevrange/(g-1)
    inp = int(2*m)
    outn = int(n)
    for jj in np.arange(1,inp+2):  # ???: Missing the last value on purpose?
        for fxn in np.arange(1,outn+1):
            if error[jj - 1][fxn - 1] > 0:
                G[jj - 1][fxn - 1] = G[jj - 1][fxn - 1] + dg*deltas[fxn - 1]
            elif error[jj - 1][fxn - 1] < 0:
                G[jj - 1][fxn - 1] = G[jj - 1][fxn - 1] - dg*deltas[fxn - 1]
                
    G[G < gmin] = gmin
    G[G > gmax] = gmax
    return G


def _gdummynvana_numpy(G, gmax, gmin, g, m, n, error, deltas):
    """Beware: this implementation does not really bother about the
    temporary copies of possibly heavy arrays...
    """
    inp = int(2 * m)
    outn = int(n)
    shape = (inp + 1, outn)

    dg = (gmax - gmin) / (g - 1.0)
    # NB: broadcasting is memory-efficient
    fine_dgs = np.broadcast_to(dg*deltas[np.newaxis, :], shape)

    # Beware that we first have to have to extract subviews of the
    # relevant arrays as we avoid the last row of G (but why do we?!)
    G_view = G[0:inp + 1, 0:outn]
    error_view = error[0:inp + 1, 0:outn]
    fine_dgs_view = fine_dgs[0:inp + 1, 0:outn]

    selection = error_view > 0
    G_view[selection] += fine_dgs_view[selection]
    selection = error_view < 0
    G_view[selection] -= fine_dgs_view[selection]

    G = np.clip(G, gmin, gmax)

    return G


jit(nopython=True)
def _gdummynvana_jit(G, gmax, gmin, g, m, n, error, deltas):

    inp = int(2 * m)
    outn = int(n)

    dg = (gmax - gmin) / (g - 1.0)

    # Switched loops order to avoid useless computations of dg*deltas[fxn}]
    for fxn in range(outn):
        fine_dg = dg * deltas[fxn]

        for jj in range(inp + 1):
            if error[jj, fxn] > 0:
                G[jj, fxn] += fine_dg
                continue # NB: comment it if optimizing the loops
            if error[jj, fxn] < 0:
                G[jj, fxn] -= fine_dg

    # NB: Genuine re-looping: may waste some time
    for row in range(G.shape[0]):
        for col in range(G.shape[1]):
            if G[row, col] < gmin:
                G[row, col] = gmin
                continue  # skip the next test for speed
            if G[row, col] > gmax:
                G[row, col] = gmax

    return G


def gdummynvana(G, gmax, gmin, g, m, n, error, deltas):
    """TODO: write a docstring!

    Parameters
    ----------
    G : ndarray
        Conductance values of the synaptic crossbar.
        Its shape is (2*m + 2, n).

    gmax, gmin : floats
        Maximal, resp. minimal, value of conductance.

    g : int
        Number of writable synaptic levels.

    m, n : int
        Number of features, resp. classes.

    error : ndarray
        ??? DESCRIPTION!

    deltas : ndarray
        ??? DESCRIPTION!

    Returns
    -------
    up-to-date G : ndarray
    """
    # Weirdly, the JIT-compiled version is not currently the fastest approach.
    return _gdummynvana_numpy(G, gmax, gmin, g, m, n, error, deltas)


#TO DO def Greal(...)


if __name__ == '__main__':
    # One avoids using `pytest` for simplicity sake but if the
    # library becomes really big, it may be worth adopting such
    # a proper unit testing library.
    #
    # At the moment, just run the file as a script and pray for nothing to
    # yell at you ;).

    print("Performing some poor man's (not so unit) testing.\n")

    # ##################
    # Helper function(s)
    def get_dummy_data(seed=None):
        _prng = np.random.RandomState(seed)  # for reproducibility
        m = 4*10  # number of features
        n = 3*10  # number classes
        gmin, gmax = 1e-8, 1e-6
        g = 256  # number of writable levels
        G = _prng.uniform(low=gmin, high=gmax, size=(2*m + 2, n))
        error = _prng.random_sample(size=G.shape)
        deltas = np.zeros(n)
        gmins = _prng.normal(loc=gmin, scale=0.2 * gmin, size=G.shape)
        gmaxs = _prng.normal(loc=gmax, scale=0.2 * gmax, size=G.shape)
        memarray = np.dstack((gmaxs, gmins))

        return G, gmax, gmin, g, m, n, error, deltas, memarray

    # #################
    # Testing `gdummyv`
    print("Testing 'gdummyv' implementations: ", end="\t")
    seed = 135

    results_A = []
    for func in (_gdummyv_vanilla, _gdummyv_numpy, _gdummyv_jit):
        G_A, _, _, g, m, n, error, _, memarray = get_dummy_data(seed)
        results_A.append(func(G_A, g, m, n, error, memarray))

    np.testing.assert_allclose(results_A[0], results_A[1])
    np.testing.assert_allclose(results_A[0], results_A[2])
    print("OK!")

    # ##################
    # Testing `gdummynv`
    print("Testing 'gdummynv' implementations: ", end="\t")
    seed = 357

    results_B = []
    for func in (_gdummynv_vanilla, _gdummynv_numpy, _gdummynv_jit):
        G_B, gmax, gmin, g, m, n, error, _, _ = get_dummy_data(seed)
        results_B.append(func(G_B, gmax, gmin, g, m, n, error))

    np.testing.assert_allclose(results_B[0], results_B[1])
    np.testing.assert_allclose(results_B[0], results_B[2])
    print("OK!")

    # #####################
    # Testing `gdummynvana`
    print("Testing 'gdummynvana' implementations: ", end="\t")
    seed = 579

    results_C = []
    for func in (_gdummynvana_vanilla, _gdummynvana_numpy, _gdummynvana_jit):
        G2, gmax, gmin, g, m, n, error, deltas, _ = get_dummy_data(seed)
        results_C.append(func(G2, gmax, gmin, g, m, n, error, deltas))

    np.testing.assert_allclose(results_C[0], results_C[1])
    np.testing.assert_allclose(results_C[0], results_C[2])
    print("OK!")

    print("\nIf you are reading this, then everything " +
          "seems to have run nicely :).")
