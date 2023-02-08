import numpy as np
import sys
from matplotlib import pyplot as plt
sys.path.insert(1,'.')
from help_fcts import j2a, j2_emp, j2a_m, opt_sigma, opt_sigma_m
from matplotlib.lines import Line2D

FS_TITLE=14
FS_LAB=12
FS_LEG=12
N=10001

n=50
p=1
noise=0.2
print(2*n*np.exp(-1.5))
np.random.seed(0)

def f(X):
  y=np.sin(2*np.pi*X[:,0])
  if X.shape[1]>1:
    for ip in range(1,p):
      y*= np.sin(2*np.pi*X[:,ip])
  return y.reshape((-1,1))

x_c=3*np.random.standard_cauchy((n,1))
x_u=np.random.uniform(-10,10,(n,1))
x_n=np.random.normal(0,10,(n,1))
x_e=np.random.exponential(10,(n,1))
TITLES=['Cauchy', 'Uniform', 'Normal','Exponential']
fig,ax_mat=plt.subplots(4,3,figsize=(10,10))
for ii,(x,title) in enumerate(zip([x_c, x_u, x_n, x_e],TITLES)):
  y=f(x)+np.random.normal(0,noise,(n,p))
  x1=np.linspace(np.min(x),np.max(x),N).reshape((-1,1))
  
  
  SIGMA_BDS=[[1e-10,.4],[0.01,1],[0.01,1]]
  LBDAS=[0, 10, 30]
  Y_LIMS_A=[[None,1e4],[None,None],[None,None]]
  Y_LIMS_E=[[None,None],[None,None],[None,None]]
  
  axs=ax_mat[ii,:]
  for ax,lbda,sigma_bds, y_lim_a, y_lim_e in zip(axs,LBDAS,SIGMA_BDS,Y_LIMS_A, Y_LIMS_E):
    sigmas=np.linspace(sigma_bds[0],sigma_bds[1],1000)
    j2a_0, j2a_1, j2a_2 = j2a_m(sigmas,x,lbda)
    D=np.abs(x-x.T)
    sigma_j=opt_sigma_m(D,lbda)[0]
    
    j2_emp_0= j2_emp(x,y,x1,lbda,sigmas)
    
    ax1=ax.twinx()
    ax.plot(sigmas,j2a_0, 'C2',linewidth=2)
    ax1.plot(sigmas,j2_emp_0,'C0', linewidth=2)
    if lbda==0:
      ax.set_yscale('log')
      ax1.set_yscale('log')
      ax.set_ylabel(title, fontsize=FS_TITLE)
    ax.axvline(sigma_j,color='k',linestyle='--')
    
    ax.set_ylim(y_lim_a)
    ax1.set_ylim(y_lim_e)
    if ii==0: ax.set_title('$\\lambda=$'+str(lbda), fontsize=FS_TITLE)
    if ii==3: ax.set_xlabel('$\\sigma$',fontsize=FS_LAB)
    
    lines=[]
    for col in ['C2','C0','k']: 
      if col=='k':
        lines.append(Line2D([0],[0],color=col,lw=2,linestyle='--'))
      else:
        lines.append(Line2D([0],[0],color=col,lw=2))
    
    fig.legend(lines, ['$J_2^a$','$J_2$','$\\sigma_0$'], loc='lower center', ncol=3, fontsize=FS_LEG)
    plt.tight_layout()
    fig.subplots_adjust(bottom=.1)
    plt.savefig('figures/synth_demo.pdf')
  
  
