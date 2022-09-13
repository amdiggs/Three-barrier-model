import numpy as np
import sys
import time
import pdb
import math
import plot_funcs as pf
import data_io as dio


seed = math.floor(time.time())
gen = np.random.default_rng(seed)

scale=604800
mod=6.0e+13*scale
KB=8.617e-5
TEMP=300.0
Beta=1 / (KB*TEMP)
T_file="DAT_FILES/Temp_Dat/T-{0:.0f}.txt"
DAT_file = "../../Zimanyi/Data/ASU-2021.xlsx"

class Dist():
    def __init__(self,m_id, pars, num_p):
        self.probs = None
        self.Evals = None
        self.num = num_p
        self.mu = pars[0]
        self.sig = pars[1]
        self.ID = m_id
        self.gen_probs()
    
    def gen_probs(self, lims = None):
        vals = gen.normal(self.mu, self.sig, 1000000)
        two_sig = self.sig*3
        if (lims == None):
            if(self.mu - two_sig < 0):
                lb = 0.08
            else:
                lb = self.mu - two_sig
            if(self.mu + two_sig > 1.36):
                #ub = 1.24
                ub = self.mu + two_sig
            else:
                ub = self.mu + two_sig
        else:
            lb, ub = lims
        c, b = np.histogram(vals, self.num, (lb, ub))
        d = b[1]-b[0]
        self.probs = c/np.sum(c)
        self.Evals = b[:-1] + d
    
        
    @property
    def vals(self):
        return self.probs, self.Evals
    
    def renormalize(self, dic):
        n_probs = []
        n_Evals = []
        for i in range(len(self.probs)):
            dic[self.ID] = [self.Evals[i], self.sig]
            if(defects(dic['t0'], dic['Eb'][0], dic['Er'][0], dic['Ed'][0]) <= 0.99):
                print(dic['t0'], dic['Eb'][0], dic['Er'][0], dic['Ed'][0])
                continue
            #elif(self.Evals[i] > dic['Eb'][0]):
            elif(defects(dic['tf'], dic['Eb'][0], dic['Er'][0], dic['Ed'][0]) >= 0.99):
                print(dic['tf'], dic['Eb'][0], dic['Er'][0], dic['Ed'][0])
                #n_probs.append(self.probs[i])
                #n_Evals.append(self.Evals[i])
                continue
            else:
                n_probs.append(self.probs[i])
                n_Evals.append(self.Evals[i])
        lb = np.min(n_Evals)
        ub = np.max(n_Evals)
        self.gen_probs(lims = [lb,ub])
        dic[self.ID] = [self.mu, self.sig]
        return 

def rate(EA):
    return np.longdouble(mod*np.exp(-EA*Beta))

def Gauss(x, u, var):
    norm = 1.0 / np.sqrt( 2 * np.pi * var)
    arg = -1*((x - u) * (x - u)) / (2 * var)
    return norm*np.exp(arg)

def defects(t,e1,e2,e3):
    A=rate(e1)
    B=rate(e2)
    C=rate(e3)
    a= A + B + C
    sq = np.longdouble(np.sqrt(a*a - 4*A*C))
    arg1 = np.longdouble(-0.5*(a -sq)*t)
    arg2= np.longdouble(-0.5*(a +sq)*t)
    Q = np.longdouble((B + C - A + sq) / (2*sq))
    W = np.longdouble((B + C - A - sq) / (2*sq))
    ex1 = np.exp(arg1, dtype=np.longdouble)
    ex2 = np.exp(arg2, dtype=np.longdouble)
    first = np.longdouble(Q*np.exp(arg1, dtype = np.longdouble))
    sec= np.longdouble(W*np.exp(arg2, dtype=np.longdouble))
    #pdb.set_trace()
    return np.longdouble(first - sec)


def rec_ave(t, p1, e1, e2, e3, i):
    if(i == 0):
        return defects(t,e1[i],e2,e3)*p1[i]
    else:
        return defects(t,e1[i],e2,e3)*p1[i] + rec_ave(t,p1, e1, e2, e3,i-1)

def EXP(dic):
    d1 = Dist( 'Eb',dic['Eb'], 20)
    d1.gen_probs(lims=[1.0,1.3])
    d1.renormalize(dic)
    p1, E1 = d1.vals
    print(np.sum(p1))
    print(p1)
    print(E1)
    d2 = Dist('Er', dic['Er'],30)
    d2.gen_probs(lims= [0.1,0.7])
    p2, E2 = d2.vals
    print(p2)
    print(E2)
    d3 = Dist('Ed', dic['Ed'],30)
    d3.gen_probs(lims = [0.1,0.7])
    p3, E3 = d3.vals
    print(p3)
    print(E3)
    tvals=np.arange(dic['t0'],dic['tf'], 1.0)
    print(np.sum(p2))
    print(np.sum(p3))
    expN = []
    for t in tvals:
        tmp = 0
        for k in range(len(p3)):
            for j in range(len(p2)):
                tmp += rec_ave(t,p1, E1, E2[j], E3[k], len(p1) -1)*p2[j]*p3[k]
        expN.append(tmp)
    return tvals, np.asarray(expN) 


def EXP2(dic):
    d1 = Dist( 'Eb',dic['Eb'], 20)
    d1.gen_probs(lims=[1.0,1.3])
    d1.renormalize(dic)
    p1, E1 = d1.vals
    mask = np.nonzero(p1)
    print(np.sum(p1))
    print(E1)
    print(p1)
    p2, E2 = Dist('Er', dic['Er'],40).vals
    p3, E3 = Dist('Ed', dic['Ed'],40).vals
    print(p2)
    print(E2)
    print(p3)
    print(E3)
    t1=np.arange(1.0,dic['tf'], 1.0)
    t2=np.arange(0.0, 1.1, 0.1)
    #tvals=np.concatenate((t2,t1))
    tvals = t1
    count = 0
    expN = []
    for t in tvals:
        tmp = 0
        for k in range(len(p3)):
            for j in range(len(p2)):
                for i in range(len(p1)):
                    tmp += defects(t,E1[i], E2[j], E3[k])*p2[j]*p3[k]*p1[i]
        expN.append(tmp)
    return tvals, np.asarray(expN)


def update_fit(dic, wtf):
    change = input("What the F*$K do you want???")
    for item in dic:
        if(change == item):
            new_vals = input("please enter the new values.")
            x= new_vals.split()
            if(len(x) ==1):
                dic[item] = float(x[0])
                return
            elif(len(x) == 2):
                dic[item] = [float(x[0]), float(x[1])]
                return
            else:
                print("OOPS!!!")
                return update_fit(dic, wtf)
    if(change == 'save'):
        val = input("what file name?")
        wtf.save(val)
        quit()
    else:
        quit()
    return


def Hand_fit(file1,file2):
    t1, n1, T1 = dio.read_dat(file1)
    killa = pf.wtf2(dio.get_dat(file2), [t1,n1], "DA")
    while(1):
        killa.draw()
    return

def gen_dat(dic, file):
    data = dio.get_dat(file)
    t1, N1 = EXP2(E_dict)
    w, da = pf.fit_da(data, t1)
    pf.check([[t1,N1, "reg"],[t1,N1/N1[0], "scaled"]])
    tw = t1 -(t1[0] -2.0)
    NN1 =  1.0 - N1/N1[0]
    NN2 =  1.0 - N1
    pf.check([[t1,NN1, "reg"],[t1,NN2, "scaled"]])
    A = np.max(NN1) - np.min(NN1)
    sc = 1.0
    print(tw)
    print(w)
    N_p = NN2*sc
    #pf.check([[t1,NN1, "reg"], [tw,N_p, "mod"]])
    mask = np.logical_and(w>=2,w<52)
    sc2 = pf.get_scale(NN1[mask], da[mask])
    Np2 = (NN1 + sc2[2])*sc2[0] + sc2[1]
    pf.check([[w,Np2, "tbm"], [w,da, "str"]])
    dio.write_dat(t1, Np2, "DAT_FILES/tbm_test5.txt", TEMP)
    pf.plot_fit(t1, Np2, TEMP,dic,fd= [w,da], dat =  data)


def SCALE(dic, file2):
    data = dio.get_dat(file2)
    t1, N1, T1 = dio.read_dat("DAT_FILES/tbm_test5.txt")
    tw = t1 -(t1[0] -2.0)
    w, da = pf.fit_da(data, t1)
    #pf.check([[t1,NN1, "reg"], [tw,N_p, "mod"]])
    mask = np.logical_and(w>=4,w<52)
    sc2 = pf.get_scale(N1[mask], da[mask])
    Np2 = (N1 + sc2[2])*sc2[0] + sc2[1]
    pf.check([[tw,Np2, "tbm"], [w,da, "str"]])
    dio.write_dat(tw, Np2, "DAT_FILES/scale.txt", TEMP)
    pf.plot_fit(tw, Np2, TEMP,dic,fd= [w,da], dat =  data)


def NoDeg(dic):
    t1, N1 = EXP2(E_dict)
    dio.write_dat(t1, N1, "DAT_FILES/NoDeg{0:.2f}.txt".format(dic['Ed'][0]), TEMP)
    pf.check([[t1, N1, "?"]])
    

def junk():
    plt_list = []
    nums = [0.40,0.58,0.67]
    labs = [0.41,0.57, 0.67]
    sc = [1.0, 0.95, 0.85]
    for e in [0,1,2]:
        degfile = ratio.format(nums[e])
        t, n, T = dio.read_dat(degfile)
        t = t[11:]
        N = n[11:]
        N= N/N[0]
        N = (sc[e]*(1.0 - N)*(2.9/0.4)) + 1.0
        plt_list.append([t,N,[labs[e],0.18]])
    return plt_list


if __name__=='__main__':
    scale = [1.18, 0.7]
    tz = 2.0
    file = "../../Zimanyi/Data/ASU-2021.xlsx"
    ratio = "DAT_FILES/NoDeg{0:.2f}.txt"
    curr_tbm = "DAT_FILES/tbm_test5.txt"
    start = time.time()
    E_dict = {'N' : 3.35, 'Eb': [1.31, 0.30], 'Er' : [0.28, 0.18], 'Ed' : [0.41, 0.18], 't0' : tz, 'tf': tz + 48.0, 'C' : 1.0}
    m_vals = junk()
    #SCALE(E_dict, file)
    pf.EXP_multi(m_vals, E_dict)
    #NoDeg(E_dict)
    #gen_dat(E_dict, file)
    #t, n, T = dio.read_dat("the_shiznit.txt")
    #pf.plot_fit(t, n, T,E_dict, dat = dio.get_dat(file))
    #t1, n1, T1 = dio.read_dat("DAT_FILES/tbm_test5.txt")
    #pf.plot_fit(t1, n1, T1,E_dict, dat = dio.get_dat(file))
    #Hand_fit(curr_tbm,file)















