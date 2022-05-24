import numpy as np
from matplotlib import pyplot as plt
import sys
from help_fcts import opt_sigma, kern_1d

def f(X):
  y=np.sin(np.pi*X)
  return y

np.random.seed(1)

FS_TITLE=14

N=1001
l=5
n=10
COLS=['C2','C1','C3']
ZORDS=[5,4,3]
SCALE_FACTOR=5
LBDAS=[0,0.1*SCALE_FACTOR]

x=np.random.uniform(-l/2,l/2,n).reshape((-1,1))
y=f(x)+np.random.normal(0,0.1,x.shape)
x1=np.linspace(-l/2,l/2,N).reshape((-1,1))


fig,axs=plt.subplots(1,2,figsize=(8,4))

for ax,lbda in zip(axs,LBDAS):
  sigma0=opt_sigma(n,1,l,lbda)[0]
  SIGMAS=[sigma0, 1/SCALE_FACTOR*sigma0, SCALE_FACTOR*sigma0]
  ax.plot(x1,f(x1),'C7--',zorder=1)
  ax.plot(x,y,'ok',markersize=8, zorder=2)
  for sigma,col,zord in zip(SIGMAS,COLS,ZORDS):
    K=kern_1d(x,x,sigma)
    K1=kern_1d(x1,x,sigma)
    y1=K1.dot(np.linalg.inv(K+lbda*np.eye(n))).dot(y)
    ax.plot(x1,y1,col,zorder=zord)
  ax.set_xlim([-3,3])
  ax.set_ylim([-2,2])
  ax.set_title('$\\lambda=$'+str(lbda), fontsize=FS_TITLE)
fig.legend(labels=['True Function', 'Observations','$\\sigma=\\sigma_0$','$\\sigma=\\frac{1}{'+str(SCALE_FACTOR)+'}\\sigma_0$','$\\sigma='+str(SCALE_FACTOR)+'\\sigma_0$'],loc='lower center', ncol=5)

plt.tight_layout()
fig.subplots_adjust(bottom=.18)
plt.savefig('figures/bandwidth_demo.pdf')

