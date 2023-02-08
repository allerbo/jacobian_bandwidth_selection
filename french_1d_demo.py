import numpy as np
import sys
from matplotlib import pyplot as plt
sys.path.insert(1,'.')
from help_fcts import j2a, j2_emp, opt_sigma
from datetime import datetime as dt
from matplotlib.lines import Line2D
import pandas as pd

FS_TITLE=14
FS_LAB=12
FS_LEG=12

temps_data = pd.read_csv('french_1d.csv', delimiter=";")

y = temps_data[['t']].values-273.15
x_temp = temps_data[['date']].values
x_temp1=list(map(lambda d: dt.strptime(str(d)[1:11],'%Y%m%d%H'),x_temp))
x=np.array(list(map(lambda d: (d-x_temp1[0]).total_seconds()/3600,x_temp1))).reshape((-1,1))

x1=np.linspace(np.min(x),np.max(x),1001).reshape((-1,1))

n=len(x)
l=np.max(x)-np.min(x)
p=1

2*n*np.exp(-1.5)



SIGMA_BDS=[[0.007,7],[1e-1,7],[1e-1,7]]
LBDAS=[0, 50, 150]
Y_LIMS_A=[[None,.6],[None,0.015],[None,0.015]]
Y_LIMS_E=[[None,6],[None,0.15],[None,0.05]]

fig,axs=plt.subplots(1,3,figsize=(10,3))

for ax,lbda,sigma_bds, y_lim_a, y_lim_e in zip(axs,LBDAS,SIGMA_BDS,Y_LIMS_A, Y_LIMS_E):
  sigmas=np.linspace(sigma_bds[0],sigma_bds[1],1000)
  sigma_j=opt_sigma(n,p,l,lbda)[0]
  j2a_0, j2a_1, j2a_2 = j2a(sigmas,n,p,l,lbda)
  
  j2_emp_0= j2_emp(x,y,x1,lbda,sigmas)
  
  ax1=ax.twinx()
  ax.plot(sigmas,j2a_0, 'C2',linewidth=2)
  ax1.plot(sigmas,j2_emp_0,'C0', linewidth=2)
  if lbda==0:
    ax.set_yscale('log')
    ax1.set_yscale('log')
  ax.axvline(sigma_j,color='k',linestyle='--')
  
  ax.set_ylim(y_lim_a)
  ax1.set_ylim(y_lim_e)
  ax.set_title('$\\lambda=$'+str(lbda), fontsize=FS_TITLE)
  ax.set_xlabel('$\\sigma$',fontsize=FS_LAB)
  
  lines=[]
  for col in ['C2','C0','k']: 
    if col=='k':
      lines.append(Line2D([0],[0],color=col,lw=2,linestyle='--'))
    else:
      lines.append(Line2D([0],[0],color=col,lw=2))
  
  fig.legend(lines, ['$J_2^a$','$J_2$','$\\sigma_0$'], loc='lower center', ncol=3, fontsize=FS_LEG)
  plt.tight_layout()
  fig.subplots_adjust(bottom=.3)
  plt.savefig('figures/french_1d_demo.pdf')



