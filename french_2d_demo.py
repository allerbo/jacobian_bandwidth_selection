import numpy as np
import sys
from matplotlib import pyplot as plt
sys.path.insert(1,'.')
from matplotlib.lines import Line2D
from help_fcts import j2a, j2_emp_map, calc_D_map, opt_sigma
import pandas as pd




FS_TITLE=14
FS_LAB=12
FS_LEG=12

fpd = pd.read_csv('french_2d.csv', delimiter=";")
lats_st = fpd.Latitude.to_numpy()
lons_st = fpd.Longitude.to_numpy()
X=np.vstack((lats_st,lons_st)).T

y = fpd[['t']].values-273.15

bound_map = ((42.3, -5), (51.1, 8.3))

# Create points for interpolation
n_width = 100 
lat_map_uni = np.linspace(bound_map[0][0], bound_map[1][0], n_width)
lon_map_uni = np.linspace(bound_map[0][1], bound_map[1][1], n_width)
lat_map, lon_map = np.meshgrid(lat_map_uni, lon_map_uni)
lat_map_ = lat_map.ravel()
lon_map_ = lon_map.ravel()
X1=np.vstack((lat_map_,lon_map_)).T

D=calc_D_map(X,X)
D1=calc_D_map(X1,X)


n,p=X.shape
l=np.max(D)


2*n*np.exp(-1.5)

SIGMA_BDS=[[1e4,2.3e5],[5e3,7e5],[5e3,7e5]]
LBDAS=[0, 5, 20]
Y_LIMS_A=[[None,2e-6],[0,1.2e-6],[0,1.2e-6]]
Y_LIMS_E=[[None,4e-4],[0,5e-5],[0,5e-5]]

fig,axs=plt.subplots(1,3,figsize=(10,3))

for ax,lbda,sigma_bds, y_lim_a, y_lim_e in zip(axs,LBDAS,SIGMA_BDS,Y_LIMS_A, Y_LIMS_E):
  sigmas=np.linspace(sigma_bds[0],sigma_bds[1],1000)
  sigma_j=0.001*opt_sigma(n,p,l,lbda)[0]
  j2a_0, j2a_1, j2a_2 = j2a(sigmas,n,p,l,lbda)
  
  j2_emp_0= j2_emp_map(D,y,D1,lat_map,lon_map,lbda,sigmas)
  
  ax1=ax.twinx()
  ax.plot(0.001*sigmas,j2a_0, 'C2',linewidth=2)
  ax1.plot(0.001*sigmas,j2_emp_0,'C0', linewidth=2)
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
  plt.savefig('figures/french_2d_demo.pdf')



