import numpy as np
import sys, os
from matplotlib import pyplot as plt
sys.path.insert(1,'.')
import glob


def plot_meanq(ax,x_ax,list_o,c,zord,quant=0.1,with_quants=True, km=False):
  mat=np.vstack(list_o)
  if km:
    mat*=0.001
  if with_quants:
    ax.plot(x_ax,np.quantile(mat,quant,1),c+'--', zorder=zord)
    ax.plot(x_ax,np.quantile(mat,1-quant,1),c+'--', zorder=zord)
  return ax.plot(x_ax,np.mean(mat,1),c, zorder=zord)

JAC_SEED=False

FS_TITLE=16
FS_LAB=12
n_rows=4
fig,ax_mat=plt.subplots(n_rows,3,figsize=(12,2.5*n_rows))
if len(ax_mat.shape)==1:
  ax_mat=np.expand_dims(ax_mat,axis=0)

QUANT=0.05
ls=[]
for i_type,(TYPE, title) in enumerate(zip(['synth','french_1d','french_2d'], ['Synthetic Data', '1D Temperature Data','2D Temperature Data'])):
  type_strs_n=glob.glob('data/'+TYPE+'_r2s_0_n_*npy')
  ns=sorted(list(map(lambda s: int(s.split('_')[-1][:-4]), type_strs_n)))
  
  r2s_0_n_o=[]
  r2s_cv_n_o=[]
  r2s_sil_n_o=[]
  sigmas_0_n_o=[]
  sigmas_cv_n_o=[]
  sigmas_sil_n_o=[]
  if JAC_SEED:
    r2s_cv1_n_o=[]
    sigmas_cv1_n_o=[]
  
  for n in ns:
    r2s_0_n=np.load('data/'+TYPE+'_r2s_0_n_'+str(n)+'.npy')
    r2s_cv_n=np.load('data/'+TYPE+'_r2s_cv_n_'+str(n)+'.npy')
    r2s_sil_n=np.load('data/'+TYPE+'_r2s_sil_n_'+str(n)+'.npy')
    sigmas_0_n=np.load('data/'+TYPE+'_sigmas_0_n_'+str(n)+'.npy')
    sigmas_cv_n=np.load('data/'+TYPE+'_sigmas_cv_n_'+str(n)+'.npy')
    sigmas_sil_n=np.load('data/'+TYPE+'_sigmas_sil_n_'+str(n)+'.npy')
    
    r2s_0_n_o.append(r2s_0_n)
    r2s_cv_n_o.append(r2s_cv_n)
    r2s_sil_n_o.append(r2s_sil_n)
    sigmas_0_n_o.append(sigmas_0_n)
    sigmas_cv_n_o.append(sigmas_cv_n)
    sigmas_sil_n_o.append(sigmas_sil_n)
    
    if JAC_SEED:
      r2s_cv1_n=np.load('data/'+TYPE+'_r2s_cv1_n_'+str(n)+'.npy')
      sigmas_cv1_n=np.load('data/'+TYPE+'_sigmas_cv1_n_'+str(n)+'.npy')
      r2s_cv1_n_o.append(r2s_cv1_n)
      sigmas_cv1_n_o.append(sigmas_cv1_n)
   
  r2s_0_lbda_o=[]
  r2s_cv_lbda_o=[]
  r2s_sil_lbda_o=[]
  sigmas_0_lbda_o=[]
  sigmas_cv_lbda_o=[]
  sigmas_sil_lbda_o=[]
  if JAC_SEED:
    r2s_cv1_lbda_o=[]
    sigmas_cv1_lbda_o=[]
  
  type_strs_lbda=glob.glob('data/'+TYPE+'_r2s_0*lbda*npy')
  log10lbdas=np.array(sorted(list(map(lambda s: int(s.split('_')[-1][:-4]), type_strs_lbda))))
  lbdas=10**(0.1*log10lbdas)
  
  for log10lbda in log10lbdas:
    r2s_0_lbda=np.load('data/'+TYPE+'_r2s_0_lbda_'+str(log10lbda)+'.npy')
    r2s_cv_lbda=np.load('data/'+TYPE+'_r2s_cv_lbda_'+str(log10lbda)+'.npy')
    r2s_sil_lbda=np.load('data/'+TYPE+'_r2s_sil_lbda_'+str(log10lbda)+'.npy')
    sigmas_0_lbda=np.load('data/'+TYPE+'_sigmas_0_lbda_'+str(log10lbda)+'.npy')
    sigmas_cv_lbda=np.load('data/'+TYPE+'_sigmas_cv_lbda_'+str(log10lbda)+'.npy')
    sigmas_sil_lbda=np.load('data/'+TYPE+'_sigmas_sil_lbda_'+str(log10lbda)+'.npy')
    
    r2s_0_lbda_o.append(r2s_0_lbda)
    r2s_cv_lbda_o.append(r2s_cv_lbda)
    r2s_sil_lbda_o.append(r2s_sil_lbda)
    sigmas_0_lbda_o.append(sigmas_0_lbda)
    sigmas_cv_lbda_o.append(sigmas_cv_lbda)
    sigmas_sil_lbda_o.append(sigmas_sil_lbda)
    
    if JAC_SEED:
      r2s_cv1_lbda=np.load('data/'+TYPE+'_r2s_cv1_lbda_'+str(log10lbda)+'.npy')
      sigmas_cv1_lbda=np.load('data/'+TYPE+'_sigmas_cv1_lbda_'+str(log10lbda)+'.npy')
      r2s_cv1_lbda_o.append(r2s_cv1_lbda)
      sigmas_cv1_lbda_o.append(sigmas_cv1_lbda)
  
  for n_row in range(n_rows):
    ax=ax_mat[n_row,i_type]
    if n_row==0:
      if i_type==0:
        ls.append(plot_meanq(ax,ns,r2s_0_n_o,'C2',3,quant=QUANT)[0])
        ls.append(plot_meanq(ax,ns,r2s_cv_n_o,'C1',2,quant=QUANT)[0])
        ls.append(plot_meanq(ax,ns,r2s_sil_n_o,'C3',1,quant=QUANT)[0])
        if JAC_SEED: ls.append(plot_meanq(ax,ns,r2s_cv1_n_o,'C0',4,quant=QUANT)[0])
      else:
        plot_meanq(ax,ns,r2s_0_n_o,'C2',3,quant=QUANT)
        plot_meanq(ax,ns,r2s_cv_n_o,'C1',2,quant=QUANT)
        plot_meanq(ax,ns,r2s_sil_n_o,'C3',1,quant=QUANT)
        if JAC_SEED: plot_meanq(ax,ns,r2s_cv1_n_o,'C0',4,quant=QUANT)
    elif n_row==1:
      plot_meanq(ax,ns,sigmas_0_n_o,'C2',3, quant=QUANT,km=(TYPE=='french_2d'))
      plot_meanq(ax,ns,sigmas_cv_n_o,'C1',2, quant=QUANT,km=(TYPE=='french_2d'))
      plot_meanq(ax,ns,sigmas_sil_n_o,'C3',1, quant=QUANT,km=(TYPE=='french_2d'))
      if JAC_SEED: plot_meanq(ax,ns,sigmas_cv1_n_o,'C0',4, quant=QUANT,km=(TYPE=='french_2d'))
    elif n_row==2:
      plot_meanq(ax,lbdas,r2s_0_lbda_o,'C2',3, quant=QUANT)
      plot_meanq(ax,lbdas,r2s_cv_lbda_o,'C1',2, quant=QUANT)
      plot_meanq(ax,lbdas,r2s_sil_lbda_o,'C3',1, quant=QUANT)
      if JAC_SEED: plot_meanq(ax,lbdas,r2s_cv1_lbda_o,'C0',4, quant=QUANT)
    elif n_row==3:
      plot_meanq(ax,lbdas,sigmas_0_lbda_o,'C2',3, quant=QUANT,km=(TYPE=='french_2d'))
      plot_meanq(ax,lbdas,sigmas_cv_lbda_o,'C1',2, quant=QUANT,km=(TYPE=='french_2d'))
      plot_meanq(ax,lbdas,sigmas_sil_lbda_o,'C3',1, quant=QUANT,km=(TYPE=='french_2d'))
      if JAC_SEED: plot_meanq(ax,lbdas,sigmas_cv1_lbda_o,'C0',4, quant=QUANT,km=(TYPE=='french_2d'))
    
    if n_row==0 or n_row==2:
      ax.set_ylim([-1,1.1])
      ax.axhline(1,color='k')
      ax.axhline(0,color='k')
    if n_row==0:
      ax.set_title(title,fontsize=FS_TITLE)
    if n_row==1 or n_row==3:
      ax.set_yscale('log')
    
    if i_type==0:
      if n_row==0 or n_row==2:
        ax.set_ylabel('$R^2$',fontsize=FS_LAB)
      elif n_row==1 or n_row==3:
        ax.set_ylabel('$\\sigma$',fontsize=FS_LAB)
        ax.set_yscale('log')
    if n_row==0 or n_row==1:
      ax.set_xlabel('n',fontsize=FS_LAB)
    elif n_row==2 or n_row==3:
      ax.set_xscale('log')
      ax.set_xlabel('$\\lambda$',fontsize=FS_LAB)

if JAC_SEED:
  fig.legend(ls, ['Jacobian','Cross-validation','Silverman', 'Jacobian Seeded Cross-validation'], loc='lower center', ncol=4)
else:
  fig.legend(ls, ['Jacobian','Cross-validation','Silverman'], loc='lower center', ncol=3)
plt.tight_layout()
fig.subplots_adjust(bottom=.09)
if JAC_SEED:
  plt.savefig('figures/sweep_all1.pdf')
else:
  plt.savefig('figures/sweep_all.pdf')

