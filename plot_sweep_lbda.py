import numpy as np
import sys, os
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import glob
from scipy.stats import trim_mean
import pickle

def plot_medq(ax,x_ax,list_o,c,zord,quant=0.9,with_quants=True, km=False, time=False):
  mat=np.vstack(list_o).T[2:,:]
  if km:
    mat*=0.001
  elif time:
    mat*=1000
  if with_quants:
    #ax.plot(x_ax,np.mean(mat,axis=1),c, zorder=zord)
    ax.plot(x_ax,trim_mean(mat,1-quant,axis=1),c, zorder=zord)
    #ax.plot(x_ax,np.median(mat,axis=1),c, zorder=zord)
    ax.plot(x_ax,np.quantile(mat,quant,1),c+'--', zorder=zord)
    ax.plot(x_ax,np.quantile(mat,1-quant,1),c+'--', zorder=zord)
    ax.fill_between(x_ax,np.quantile(mat,quant,1),np.quantile(mat,1-quant,1),color=c, zorder=zord, alpha=0.1)



SUFS=['french_2d', 'french_1d', 'synth_c']
TITLES=['2D Temperature Data','1D Temperature Data', 'Cauchy Distribution']
LBDASS=[np.logspace(-4,2,13),np.logspace(-4,2,13),np.logspace(-4,2,13)]
METRICS=['r2','sigma','time']
Y_LABS=['$R^2$','$\\sigma$','$t$']
ALGS=['j','gcv','lm','sil']
COLS=['C2','C1','C0','C3']
Z_ORDS=[4,3,1,0]

plot_dict={}
for suf in SUFS:
  plot_dict[suf]={}
  for metric in METRICS:
    plot_dict[suf][metric]={}
    for alg in ALGS:
      if suf=='synth_c' and alg=='j': alg='jm'
      plot_dict[suf][metric][alg]=[]
  
  type_strs_lbda=glob.glob('data/'+suf+'_lbda_*.pkl')
  seeds=sorted(list(map(lambda s: int(s.split('_')[-1][:-4]), type_strs_lbda)))
  for seed in seeds:
    fi=open('data/'+suf+'_lbda_'+str(seed)+'.pkl','rb')
    data_dict=pickle.load(fi)
    fi.close()
    for alg in ALGS:
      if suf=='synth_c' and alg=='j': alg='jm'
      for metric in METRICS:
        plot_dict[suf][metric][alg].append(data_dict[alg][metric])

fig,ax_mat=plt.subplots(3,3,figsize=(12*.8,7.5*.8))
for ac, (suf,title,lbdas) in enumerate(zip(SUFS,TITLES,LBDASS)):
  for metric,y_lab,ax in zip(METRICS,Y_LABS,ax_mat[:,ac]):
    for alg,col,z_ord in zip(ALGS,COLS,Z_ORDS):
      if suf=='synth_c' and alg=='j': alg='jm'
      plot_medq(ax,lbdas,plot_dict[suf][metric][alg],col,z_ord,km=(metric=='sigma' and suf=='french_2d'),time=(metric=='time'))
      if metric=='r2':    
        ax.axhline(0,color='black')
    
    if ac==0:
      ax.set_ylabel(y_lab)
    if metric=='r2':
      ax.set_title(title)
      ax.set_ylim([-.3,1.1])
    if metric=='sigma':
      ax.set_yscale('log')
      if suf=='french_1d':
        ax.set_ylim([1,None])
        ax.set_yticks([1e0,1e1,1e2,1e3])
      elif suf=='french_2d': 
        ax.set_ylim([40,None])
        ax.set_yticks([1e2,1e3])
    if metric=='time':
      ax.set_xlabel('$\\lambda$')
      ax.set_yscale('log')
      #ax.set_yticks([1e-5,1e-3,1e-1,10])
      #ax.set_ylim([1e-5,1e2])
      ax.set_yticks([1e-2,1,1e2,1e4])
      ax.set_ylim([1e-2,1e5])
    ax.set_xscale('log')

lines=[]
for col in COLS:
  lines.append(Line2D([0],[0],color=col,lw=2))

fig.legend(lines, ['Jacobian','GCV','MML', 'Silverman'], loc='lower center', ncol=4)
  
plt.tight_layout()
fig.subplots_adjust(bottom=.13)
plt.savefig('figures/sweep_lbda.pdf')

