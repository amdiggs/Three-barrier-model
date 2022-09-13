import numpy as np
import matplotlib.pyplot as plt
import sys
import pdb
from scipy.optimize import curve_fit as cf
#1 read dat files
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

def compare(N1,t1, N2, t2):
    cont1 = []
    cont2 =[]
    L1=len(N1)
    L2=len(N2)
    st=0
    prev= 1000.0
    for i in range(L1):
        v1=N1[i]
        #pdb.set_trace()
        for j in range(st,L2):
            v2=N2[j]
            ratio = abs(1.0 - v1 / v2)
            if( ratio > prev):
                prev=1000.0
                break
            elif(ratio <= 0.000005):
                #print(v1,v2)
                #print(i, j)
                cont1.append(t1[i])
                cont2.append(t2[j])
                prev = 1000.0
                break
            else:
                prev=ratio
                continue
    return np.asarray(cont1), np.asarray(cont2)

def line(x,m,b):
    return x*m +b

def power_law(x,b, c):
    return np.power(x,b) + c


def plot_time_corr(TEMP1,TEMP2):
    fig = plt.figure(figsize=[10,6])
    ax = fig.add_subplot()
    sc=604800.0
    file = "DAT2/N_T{0:.2f}.txt"
    t1, N1, T1 = read_dat(file.format(TEMP1))
    t2, N2, T2 = read_dat(file.format(TEMP2))
    tm1, tm2 = compare(N1[1:],t1,N2[1:], t2)
    tm1= (tm1)*sc
    tm2 = (tm2)*sc
    par, cov = cf(power_law, tm1,tm2)
    x= np.arange(np.min(tm1),np.max(tm1), 1000)
    y=power_law(x,*par)
    print(par) 
    a, b = np.log(tm1), np.log(tm2)
    p2, c2 = cf(line, a, b)
    print(p2)
    x2=np.arange(np.min(a), np.max(a))
    ax.scatter(tm1,tm2)
    ax.plot(x, y, "r--", label="Fit")
    ax.set_xlabel("Accelerated time (s)")
    ax.set_ylabel("Normal time (s)")
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #ax.set_xlim(1.0,1.0e+8)
    #ax.set_ylim(1,1.0e+9)
    plt.title("Time Correspondence for T = {0:.0f} K, T = {1:.0f} K".format(T1,T2))
    plt.show()





if __name__=='__main__':
    plot_time_corr(325.0, 380.0)





















