import numpy as np
import sys, os
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import glob
from scipy.stats import trim_mean

def plot_medq(ax,x_ax,list_o,c,zord,quant=0.2,with_quants=True, km=False):
  mat=np.vstack(list_o)
  if km:
    mat*=0.001
  if with_quants:
    ax.plot(x_ax,trim_mean(mat,quant,axis=1),c, zorder=zord)
    ax.plot(x_ax,np.quantile(mat,quant,1),c+'--', zorder=zord)
    ax.plot(x_ax,np.quantile(mat,1-quant,1),c+'--', zorder=zord)
    ax.fill_between(x_ax,np.quantile(mat,quant,1),np.quantile(mat,1-quant,1),color=c, zorder=zord, alpha=0.1)



SUFS=['french_2d', 'french_1d', 'synth_c']
TITLES=['2D Temperature Data', '1D Temperature Data', 'Cauchy Distribution']
METRICS=['r2','sigma']
Y_LABS=['$R^2$','$\\sigma$']
ALGS=['jcv','cv', 'jlm', 'lm']
COLS=['C2','C1', 'C0', 'C4']
Z_ORDS=[4,3,2,1]

fig,ax_mat=plt.subplots(2,3,figsize=(9,4))
for ac, (suf,title) in enumerate(zip(SUFS,TITLES)):
  for metric,y_lab,ax in zip(METRICS,Y_LABS,ax_mat[:,ac]):
    type_strs_n=glob.glob('data_'+suf+'/'+suf+'_'+metric+'s_jcv_n_*.npy')
    ns=sorted(list(map(lambda s: int(s.split('_')[-1][:-4]), type_strs_n)))
    for alg,col,z_ord in zip(ALGS,COLS,Z_ORDS):
      data=[]
      for n in ns:
        data.append(np.load('data_'+suf+'/'+suf+'_'+metric+'s_'+alg+'_n_'+str(n)+'.npy'))
      plot_medq(ax,ns,data,col,z_ord,km=(metric=='sigma' and suf=='french_2d'))
      if metric=='r2':    
        ax.axhline(0,color='black')
    
    if ac==0:
      ax.set_ylabel(y_lab)
    if metric=='r2':
      ax.set_title(title)
      ax.set_ylim([-.3,1.1])
    if metric=='sigma':
      ax.set_yscale('log')
      if suf=='synth_c':
        ax.set_yticks([1e-3,1e-2,1e-1,1e0,1e1])
      elif suf=='french_1d':
        ax.set_ylim([1,None])
        ax.set_yticks([1e0,1e1,1e2,1e3])
      elif suf=='french_2d': 
        ax.set_yticks([1e2,1e3])
        ax.set_yticklabels([],minor=True)
      ax.set_xlabel('n')
  
lines=[]
for col in COLS:
  lines.append(Line2D([0],[0],color=col,lw=2))

fig.legend(lines, ['Jacobian Seeded 10-fold CV','10-fold Cross-Validation','Jacobian Seeded Log Marginal','Log Marginal'], loc='lower center', ncol=2)
  
plt.tight_layout()
fig.subplots_adjust(bottom=.25)
plt.savefig('figures/sweep_n_seeded.pdf')

