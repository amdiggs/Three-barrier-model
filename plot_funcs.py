import numpy as np
import sys
import time
import pdb
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt
import data_io as dio
from matplotlib import cm
plt.style.use('seaborn-deep')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['mediumblue', 'crimson','darkgreen', 'darkorange','crimson', 'darkorchid'])
plt.rcParams['figure.figsize'] = [12,10]
plt.rcParams['axes.linewidth'] = 1.7
plt.rcParams['lines.linewidth'] = 6.0
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 22
plt.rcParams['font.family'] =  'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
#plt.rcParams['axes.facecolor'] = 'grey'
plt.ion()


class TRANSFER():
    
    def __init__(self, probs, evs):
        self.fig = plt.figure(figsize=[10,10])
        self.ax = self.fig.add_subplot()
        self.time = 0.0
        self.Nvals = None
        self.Evals = evs
        self.pmf = probs
        self.ub = np.max(probs)*(1.1)
    
    def set_N(self, arr):
        self.Nvals = arr
        return
    
    def set_time(self, t):
        self.time= t
        return
    
    def get_fit(self):
        par, cov = opt.curve_fit(exp_dist, self.Evals, self.Nvals, p0 = [1.36, 0.4, -1.0])
        x = np.arange(0.5,par[0],0.1 )
        y = exp_dist(x, *par)
        return x, y, par
    
    def draw(self):
        #pdb.set_trace()
        #x, y, par = self.get_fit()
        txt= "t = {0:.3f} weeks".format(self.time)
        #p_text = r"$\lambda $ = {0:.3f}".format(par[0])
        delta = self.Evals[1] - self.Evals[0]
        self.ax.bar(self.Evals, self.Nvals, width = delta, edgecolor='white')
        #self.ax.plot(x,y,color =  "crimson", label="Exponential Distribution")
        self.ax.text(0.8, 0.94, txt, transform = self.ax.transAxes)
        #self.ax.text(0.05, 0.88, p_text, transform = self.ax.transAxes)
        self.ax.set_ylim(0.0,self.ub)
        self.ax.set_xlabel("Barrier Energy (eV)")
        self.ax.set_ylabel("P(E(bond))")
        plt.title("Distribution of Si-H Bond Energies")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.fig.savefig("PEB_t{0:.0f}.jpg".format(self.time))
        plt.waitforbuttonpress()
        #plt.show(block=False)
        #plt.pause(0.1)
        self.ax.clear()
        return

class WTF(object):
   
    def __init__(self,DAT, dlabel):
        self.fig = plt.figure(figsize=[10,6])
        self.ax = self.fig.add_subplot()
        self.dat = DAT
        self.lab = dlabel
        self.time = None
        self.Nvals = None
        self.fit_vals = None
        self.fit_dat()
    
    @property
    def vals(self):
        return self.time, self.Nvals
    
    @vals.setter
    def vals(self, arr):
        self.time, self.Nvals = arr
        return
    def save(self, name):
        self.fig.savefig("fit_{0}.png".format(name))
        return
    
    def fit_dat(self):
        guess = [3.5, 12.0, 1.0]
        par, cov = opt.curve_fit(decay, self.dat[0], self.dat[1], p0 = guess)
        x= np.arange(np.min(self.dat[0]), np.max(self.dat[0]), 1.0)
        y = decay(x, *par)
        self.fit_vals = [x,y]
        return
    
    def draw(self, dic):
        #E_dict = {'N' : 20.6, 'Eb': [1.36, 0.32], 'Er' : [0.21, 0.18], 'Ed' : [0.39, 0.15], 't0' : 1.0, 'tf': 52.0}
        #pdb.set_trace()
        self.ax.clear()
        #rat= dic['e3']/dic['e2']
        #rat_txt = r'$\frac{K_{forward}}{K_{reverse}} $ = ' + "{0:.3f}".format(rat)
        #txt = "N = {0:.2e}\nEA2 = {1:.3f}\n EA3 = {2:.3f}".format(dic['N'], dic['e2'], dic['e3'])
        #d_txt = "EA1 Normal Distribution\n" +r'$\mu $= {0:.2f} $\sigma $ = {1:.3f}'.format(dic['mu'], dic['sig'])
        NN1 = 1.0 - self.Nvals
        A = np.max(NN1) - np.min(NN1)
        sc = dic['N']/A
        N_p = NN1*sc + dic['C']
        td = np.min(self.dat[0]) - dic['t0']
        t_p = self.time + td
        self.ax.plot(t_p, N_p, label = "TBM")
        self.ax.scatter(self.dat[0],self.dat[1], label = "Data")
        self.ax.plot(self.fit_vals[0], self.fit_vals[1], label = "Exp Fit")
        #self.ax.text(0.5, 0.5, txt, transform=self.ax.transAxes)
        #self.ax.text(0.1, 0.9, rat_txt, transform=self.ax.transAxes)
        #self.ax.text(0.5, 0.35, d_txt, transform=self.ax.transAxes)
        self.ax.set_xlabel("t (weeks)")
        self.ax.set_ylabel("N")
        self.ax.set_ylim(0.8,5.0)
        plt.title("Three Barrier Model fit to {0} data".format(self.lab))
        plt.legend()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.waitforbuttonpress(1)
        return
    

class wtf2(object):
   
    def __init__(self,DAT, DAT2, dlabel):
        self.fig = plt.figure(figsize=[12,10])
        self.ax = self.fig.add_subplot()
        self.dat = DAT
        self.lab = dlabel
        self.dat2 = DAT2
        self.A = 1.0
        self.const = 0.0
        self.off_set = 0.0
    
    @property
    def vals(self):
        return self.dat2[0]- self.off_set, self.dat2[1]*self.A + self.const
    
    def save(self, name):
        self.fig.savefig("fit_{0}.png".format(name))
        quit()
    
    def update_params(self):
        choice = input("Which param would you like to up date? ")
        if(choice == 'a'):
            value = input("Enter new value: ")
            self.A = float(value)
            return
        elif(choice == 'c'):
            value = input("Enter new value: ")
            self.const = float(value)
            return
        elif(choice == 'o'):
            value = input("Enter new value: ")
            self.off_set = float(value)
            return
        elif(choice == 'save'):
            value = input("Enter filename: ")
            dio.write_dat(self.vals[0], self.vals[1],value, 300.0)#self.save(value)
        elif(choice == "q"):
            quit()
        else:
            self.update_params()
    
    
    def draw(self):
        #pdb.set_trace()
        self.ax.clear()
        bds = [[-100.0,0,0.0,-100.0], [100.0,10000.0 , 1.0, 100]]
        w1, n1 = self.dat
        w2, n2 = self.vals
        par, cov = opt.curve_fit(str_exp, w2, n2, p0= [5.0, 10.0, 0.8, 0.0],bounds = bds, maxfev= 10000000)
        y=str_exp(w2, *par)
        self.ax.scatter(w1, n1, label = "Data")
        self.ax.plot(w2,n2,label = "SolDeg")
        self.ax.plot(w2,y, "r--")
        txt="Str Exp\n" + r"$ \tau_0 $ = {0:.2f}" + "\n"+ r"$ \beta $ = {1:.2f}"
        self.ax.text(0.05, 0.65, txt.format(par[1], par[2]),fontsize = 18, transform=self.ax.transAxes)
        #self.ax.text(0.5, 0.5, txt, transform=self.ax.transAxes)
        #self.ax.text(0.1, 0.9, rat_txt, transform=self.ax.transAxes)
        #self.ax.text(0.5, 0.35, d_txt, transform=self.ax.transAxes)
        self.ax.set_xlabel("t (weeks)")
        self.ax.set_ylabel("Relative Defects")
        #self.ax.set_ylim(0.8,5.0)
        plt.title("SolDeg")
        plt.legend()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.waitforbuttonpress(1)
        self.update_params()
        return

def str_exp(t,N, tau,b, C):
    arg = -1.0*( np.power(t/tau, b))
    return N*(1.0 - np.exp(arg)) +C

def str_exp2(t,N, tau,b, C):
    arg = -1*( np.power(t/tau, b))
    return N*np.exp(arg) + C

def decay(t, N, tau, C):
    arg = -1.0*( t / tau)
    return N*(1.0 - np.exp(arg)) + C

def Gauss(x, u, var):
    norm = 1.0 / np.sqrt( 2 * np.pi * var)
    arg = -1*((x - u) * (x - u)) / (2 * var)
    return norm*np.exp(arg)

def scale(x,A, B, C):
    return A*(x + C) + B

def exp_dist(x,u,lam, A):
    arg = -1.0*(x - u)*lam
    return A*lam*np.exp(arg)

def get_scale(tb, da):
    par, cov = opt.curve_fit(scale, tb, da, p0 =[1.0, 0.0, 0.0])
    print(par)
    return par

def plot_N(tvals, N_vals, elabs, e2, params, weights):
    tvals = tvals - np.min(tvals)
    fig = plt.figure()
    ax = fig.add_subplot()
    txt= "{0} Normal Distributed\n $ \mu $ = {1:.3f}\n $\sigma^2 $ = {2:.3f}".format("Ea1", *params)
    txt2= " Ea2 = {0:.2f}".format(e2)
    itr = 0
    ave = np.zeros([len(tvals)])
    for n in N_vals:
        ave = ave + n*weights[itr]
        ax.plot(tvals, n, label = "EA3 = {0:.2f}".format(elabs[itr]))
        itr+=1
    y = str_exp2(tvals,np.max(N_vals[0]), 7.095, 0.709, 0.0)
    ax.plot(tvals, y - np.min(y), "m--")
    ax.text(0.5, .64, txt, transform =ax.transAxes)
    ax.text( 0.5, .6, txt2, transform = ax.transAxes)
    ax.set_xlabel("t (weeks)")
    ax.set_ylabel("Defects ")
    plt.title("<N> for P(Ea1)")
    plt.legend()
    #plt.savefig("PLOTS/N_nor_t{0:.1f}.png".format(np.min(tvals)))
    plt.show(block = True)

def fit_da(dat, t):
    #t = np.arange(dic['t0'],dic['tf'], 1.0)
    bds2 = [[0,0,0,-5.0], [400.0, 1000.0, 100.0,1000]]
    par, cov = opt.curve_fit(str_exp, dat[0][3:], dat[1][3:], p0=[3.500, 20.0, 0.7, 1.0],bounds = bds2, maxfev=10000000)
    y=str_exp(t, *par)
    t2 = np.arange(0.8, 46, 0.1)
    y2 = str_exp(t2, *par)
    print(par)
    txt="Str Exp\n" + r'$ \tau_0 $ = {0:.3f}'.format(par[1]) + "\n"+ r'$\beta $ = {0:.3f}'.format(par[2])
    plt.scatter(dat[0], dat[1], marker= '^',color='darkorchid', label = "DA")
    plt.plot(t2,y2, label = "Stretched Exponential")
    plt.text(24.0, 2.1, txt, fontsize = 14)
    plt.xlabel("t (weeks)")
    plt.ylabel("Relative Number of Defects")
    plt.title("Fit of Stretched Exponential to DA Sample")
    plt.legend()
    #plt.savefig("PLOTS/DA_STR_EXP.png")
    #plt.show(block = True)
    return t, y

def comp_plot(Tlist):
    file = "DAT2/N_T{0:.2f}.txt"
    fig = plt.figure()
    ax = fig.add_subplot()
    for T in Tlist:
        t1, n1, temp1  = read_dat(file.format(T))
        plt.plot(t1, n1, label ="T = " +str(T))
    t1, n1, temp1  = read_dat(file.format(300.0))
    x=[np.min(t1), np.max(t1)]
    y=[np.min(n1), np.min(n1)]
    ax.plot(x,y, "m--")
    ax.set_ylabel("N")
    ax.set_xlabel("t (weeks)")
    plt.title("[H] for Different Temperature ")
    ax.legend()
    fig.savefig("PLOTS/Tcomp.png")
    plt.show()



def plot_fit(tvals, Nvals,TEMP,dic,fd = None, dat = None):
    fig = plt.figure()
    ax = fig.add_subplot()
    sec = 604800.0
    txt_sig = r"$\sigma $ = 0.0"
    txt="Str Exp\n" + r"$ \tau_0 $ = {0:.2f}" + "\n"+ r"$ \beta $ = {1:.2f}"
    txt2="{0}\n" + r'$\mu $ ={1:.2f}' + "\n" + r'$\sigma $ = {2:.2f}'
    EB = txt2.format("E(bond)", *dic['Eb'])
    ER = txt2.format("E(recap)", *dic['Er'])
    ED = txt2.format("E(drift)", *dic['Ed'])
    if(dat != None):
        ax.scatter(dat[0], dat[1],marker = '^',color = 'darkcyan',s=320, label= "Data")
    if( fd != None):
        xxxx=0
        #ax.plot(fd[0],fd[1] ,linestyle = '--',color = (.18,.18,.31), label="Stretched Exponential Fit to Data")
    p_delta = 10.68*0.2
    bds = [[-100.0,0,0.0,0.0], [100.0,1000.0 , 1.0, 100]]
    par, cov = opt.curve_fit(str_exp, tvals, Nvals, p0= [10.0, 10.0, 0.15, 1.0],bounds = bds, maxfev= 10000000)
    y=str_exp(tvals, *par)
    mask2 = (Nvals>0.0)
    mask3 = (tvals< 47)
    wmask = np.logical_and(mask2,mask3)
    plot_t = tvals[wmask]
    plot_N = Nvals[wmask]
    plot_y = y[wmask]
    #plot_N[0] = 0.89
    #plot_t2 = plot_t
    #plot_t2[0] = 2.1
    ax.plot(plot_t - 0.3 , plot_N, label = "SolDeg")
    ax.plot(plot_t,plot_y,linestyle = '-.',linewidth = 4.0, color = 'crimson', label="Stretched Exponential Fit")
    #ax.set_xscale("log")
    ax.set_xlabel("t (weeks)")
    ax.set_ylabel("Relative Number of Defects")
    #ax.text(0.40, 0.38, txt_sig,fontsize = 20, transform=ax.transAxes)
    ax.text(0.40, 0.38, EB,fontsize = 18, transform=ax.transAxes)
    ax.text(0.60, 0.38, ER,fontsize = 18, transform=ax.transAxes)
    ax.text(0.80, 0.38, ED, fontsize = 18,transform=ax.transAxes)
    txt="Str Exp\n" + r"$ \tau_0 $ = {0:.2f}" + "\n"+ r"$ \beta $ = {1:.2f}"
    ax.text(0.05, 0.65, txt.format(par[1], par[2]),fontsize = 18, transform=ax.transAxes)
    #ax.set_xticks(np.arange(0,48,4.0))
    plt.legend(fontsize = 18)
    plt.savefig("SolDeg_fit.png")
    plt.show(block = True)


def EXP_multi(vals, dic):
    fig = plt.figure()
    ax = fig.add_subplot()
    txt= "E(rcap)\n" + r" $ \bar{E} $" + " = {1:.2f}\n $\sigma $ = {2:.2f}".format(1,*dic['Er'])
    txt2= "E(bond)\n " + r" $ \bar{E} $" +  " = {1:.2f}\n $\sigma $ = {2:.2f}".format(1, *dic['Eb'])
    for x,y, lab in vals:
        ax.plot(x,y,label = "E(drift): " + r" $\bar{E} $ " + "= {0:.2f} $\sigma $ = {1:.2f}".format(*lab))
    ax.text(0.42, 0.45, txt2, transform = ax.transAxes)
    ax.text(0.60, 0.45, txt, transform = ax.transAxes)
    ax.arrow(46.0, 3.47, 0.0, -1.67, width = 1.25, head_length = 0.25, edgecolor='darkcyan', facecolor='darkcyan')
    ax.set_xlabel("time (weeks)")
    ax.set_ylabel("Relative Number of Defects")
    plt.legend(fontsize = 18)
    fig.savefig("NoDeg.png")
    plt.show(block=True)


def check(vals):
    fig = plt.figure()
    ax = fig.add_subplot()
    for x,y, lab in vals:
        ax.plot(x,y, label = lab)
    plt.legend()
    plt.show(block=True)


def da_real(dat1, dat2):
#[ 2.  4.  6.  8. 12. 14. 17. 21. 24. 26. 30. 33. 39. 45.]
#[ 2.  4.  6.  8. 12. 14. 17. 21. 24. 26. 28. 30. 33. 36. 39. 42. 45.]
    w1, n = dat1
    w2, t = dat2
    mask = np.isin(w2, w1)
    #mask = np.logical_and(mask1, mask2)
    tau = 2.0 - t
    print(tau[0])
    print(n[0])
    s = get_scale(tau[mask][10:],n[10:])
    plt.scatter(w1,n)
    plt.scatter(w2, tau*s[0] +s[1])
    plt.show(block = True)

def FU(dat, tau):
    x=np.arange(0.0,46.0,1.0)
    y = decay(x,3.2,tau,1.0)
    plt.scatter(dat[0],dat[1])
    plt.plot(x, y)
    plt.show(block = True)



































