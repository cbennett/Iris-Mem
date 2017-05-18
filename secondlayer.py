import numpy as np

def gdummyv(G, g, m, n, error,memarray):
    inp = int(2*m)
    outn = int(n)
    for jj in np.arange(1, inp+2):
        for fxn in np.arange(1,outn +1):
            Gmax = memarray[int(jj)-1,int(fxn)-1,0]
            Gmin = memarray[int(jj)-1,int(fxn)-1,1]
            ndevrange = Gmax- Gmin
            dg = ndevrange/(g-1)
            if error[int(jj)-1][int(fxn)-1] > 0:
                G[int(jj)-1][int(fxn)-1] = G[int(jj)-1][int(fxn)-1] + dg
            elif error[int(jj)-1][int(fxn)-1] < 0:
                G[int(jj)-1][int(fxn)-1] = G[int(jj)-1][int(fxn)-1] - dg
             
            if G[int(jj)-1][int(fxn)-1] >  Gmax:
               G[int(jj)-1][int(fxn)-1] = Gmax
            elif G[int(jj)-1][int(fxn)-1] <  Gmin:
                G[int(jj)-1][int(fxn)-1] = Gmin

    return G

def gdummynvana(G, gmax, gmin, g, m, n, error,deltas):
    ndevrange = gmax- gmin
    dg = ndevrange/(g-1)
    inp = int(2*m)
    outn = int(n)
    for jj in np.arange(1,inp+2):
        for fxn in np.arange(1,outn+1):
            if error[int(jj)-1][int(fxn)-1] > 0:
                G[int(jj)-1][int(fxn)-1] = G[int(jj)-1][int(fxn)-1] + dg*deltas[int(fxn)-1]
            elif error[int(jj)-1][int(fxn)-1] < 0:
                G[int(jj)-1][int(fxn)-1] = G[int(jj)-1][int(fxn)-1] - dg*deltas[int(fxn)-1]
                
    G[G < gmin] = gmin
    G[G > gmax] = gmax
    return G

def gdummynv(G, gmax, gmin, g, m, n, error):
    ndevrange = gmax- gmin
    dg = ndevrange/(g-1)
    inp = int(2*m)
    outn = int(n)
    for jj in np.arange(1,inp+2):
        for fxn in np.arange(1,outn+1):
            if error[int(jj)-1][int(fxn)-1] > 0:
                G[int(jj)-1][int(fxn)-1] = G[int(jj)-1][int(fxn)-1] + dg
            elif error[int(jj)-1][int(fxn)-1] < 0:
                G[int(jj)-1][int(fxn)-1] = G[int(jj)-1][int(fxn)-1] - dg
                
    G[G < gmin] = gmin
    G[G > gmax] = gmax
    return G





#TO DO def Greal(...)