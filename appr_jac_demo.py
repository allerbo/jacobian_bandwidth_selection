import numpy as np
import sys
from matplotlib import pyplot as plt
sys.path.insert(1,'.')
from help_fcts import j2a

FS_TITLE=14
FS_LAB=13

np.random.seed(1)
N=1001
n=10
p=1
l=1
lbda=3


LBDAS=[0, 1, 5]
SIGMA_BDS=[[1e-11,.4],[0.01,1],[0.02,1]]

ls=[]
fig,axs=plt.subplots(1,3,figsize=(10,3.7))

for ax,lbda,sigma_bds in zip(axs,LBDAS,SIGMA_BDS):
  sigmas=np.linspace(sigma_bds[0],sigma_bds[1],1000)
  j2a_0, j2a_1, j2a_2 = j2a(sigmas,n,p,l,lbda)
  ax1=ax.twinx()
  if lbda==0:
    ls.append(ax.plot(sigmas,j2a_0,'C2', linewidth=2)[0])
    ls.append(ax1.plot(sigmas,j2a_1*0.01,'C1--', linewidth=1.5)[0])
    ls.append(ax1.plot(sigmas,j2a_2,'C3--', linewidth=1.5)[0])
    ax.set_yscale('log')
    ax1.set_yscale('log')
  else:
    ax.plot(sigmas,j2a_0, 'C2',linewidth=2)
    ax1.plot(sigmas,j2a_1*0.01,'C1--', linewidth=1.5)
    ax1.plot(sigmas,j2a_2,'C3--', linewidth=1.5)
  ax.set_title('$\\lambda=$'+str(lbda), fontsize=FS_TITLE)
  ax.set_xlabel('$\\sigma$',fontsize=FS_LAB)

fig.legend(ls, ['$J_2^a(\\sigma)$','$0.01\\cdot j_a(\\sigma)$','$j_b(\\sigma)$'], loc='lower center', ncol=3, fontsize=FS_LAB)
plt.tight_layout()
fig.subplots_adjust(bottom=.26)
plt.savefig('figures/appr_jac_demo.pdf')

