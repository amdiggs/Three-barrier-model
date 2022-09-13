import numpy as np
import pandas as pd



def read_xl(file):
    dat = pd.read_excel(file,header=[0,1]).to_numpy().T
    orig = pd.read_excel(file)
    mask = np.isfinite(dat) & np.not_equal(dat,0)
    tau = []
    ns =[]
    for i in range(1,5):
        tau.append(np.asarray([dat[0][mask[i]],dat[i][mask[i]]*1000])[:,1:])
        ns.append(np.asarray([dat[0][mask[i + 4]],dat[i + 4][mask[i + 4]]])[:,1:])
    return tau, ns

def get_dat(file, typ = 'n'):
    t, n = read_xl(file)
    if(typ == 'n'):
        w, tmp = n[0]
    else:
        w, tmp = t[0]
    da = tmp/tmp[0]
    return [w , da]

def read_dat(file):
    t =[]
    n =[]
    emax =0 
    with open(file) as f:
        for l in f.readlines():
            x = l.split()
            if(len(x) == 1):
                emax = float(x[0])
            else:
                t.append(x[0])
                n.append(x[1])
    return np.asarray(t, dtype=float), np.asarray(n, dtype= float), emax

def read_pmf(file):
    evals =[]
    pmf =[]
    with open(file) as f:
        for l in f.readlines():
            x = l.split()
            pmf.append(x[0])
            evals.append(x[1])
    return np.asarray(pmf, dtype=float), np.asarray(evals, dtype= float)



def write_dat(t,N, name, param):
    outfile = name.format(param)
    ln = "{0:.5f} {1:.6f}\n"
    with open(name, 'w') as f:
        f.write(str(param) +"\n")
        for i in range(len(t)):
             f.write(ln.format(t[i],N[i]))
    return

if __name__=='__main__':
    file = "../Zimanyi/Data/ASU-2021.xlsx"
    t, n = read_xl(path + file)
    w, tmp = n[0]
    da = tmp/np.min(tmp)
