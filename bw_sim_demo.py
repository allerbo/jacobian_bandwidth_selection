import numpy as np
import sys
from matplotlib import pyplot as plt
sys.path.insert(1,'.')
from help_fcts import j2a, j2_emp, j2a_m, opt_sigma, opt_sigma_m, calc_D_map, j2_emp_map
from matplotlib.lines import Line2D
from datetime import datetime as dt
import pandas as pd

FS_TITLE=14
FS_LAB=12
FS_LEG=12

ALGS=['2d','1d','c']
TITLES=['2D Temperature Data','1D Temperature Data', 'Cauchy Distribution']
#ALGS=['c']
#TITLES=['1']

sigma_maxs=[]

fig,axs_m=plt.subplots(3,3,figsize=(13,10))

for r_axs, (alg, title) in enumerate(zip(ALGS,TITLES)):
  if alg=='2d':
    fpd = pd.read_csv('french_2d.csv', delimiter=";")
    lats_st = fpd.Latitude.to_numpy()
    lons_st = fpd.Longitude.to_numpy()
    X=np.vstack((lats_st,lons_st)).T
    
    y = fpd[['t']].values-273.15
    
    bound_map = ((42.3, -5), (51.1, 8.3))
    
    # Create points for interpolation
    n_width = 100 
    #n_width = 10
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
    sigma_maxs.append(2*n*np.exp(-1.5))
    
    SIGMA_BDS=[[1e4,2.3e5],[5e3,7e5],[5e3,7e5]]
    LBDAS=[0, 5, 20]
    Y_LIMS_A=[[None,2e-6],[0,1.2e-6],[0,1.2e-6]]
    Y_LIMS_E=[[None,4e-4],[0,5e-5],[0,5e-5]]
  elif alg=='1d':
    temps_data = pd.read_csv('french_1d.csv', delimiter=";")
    
    y = temps_data[['t']].values-273.15
    x_temp = temps_data[['date']].values
    x_temp1=list(map(lambda d: dt.strptime(str(d)[1:11],'%Y%m%d%H'),x_temp))
    x=np.array(list(map(lambda d: (d-x_temp1[0]).total_seconds()/3600,x_temp1))).reshape((-1,1))
    
    x1=np.linspace(np.min(x),np.max(x),1001).reshape((-1,1))
    
    n=len(x)
    l=np.max(x)-np.min(x)
    p=1
    
    sigma_maxs.append(2*n*np.exp(-1.5))
    
    SIGMA_BDS=[[0.007,7],[1e-1,7],[1e-1,7]]
    LBDAS=[0, 50, 150]
    Y_LIMS_A=[[None,.6],[None,0.015],[None,0.015]]
    Y_LIMS_E=[[None,6],[None,0.15],[None,0.05]]
  elif alg=='c':
    n=50
    p=1
    noise=0.2
    N=10001
    #N=101
    np.random.seed(0)
    
    def f(X):
      y=np.sin(2*np.pi*X[:,0])
      if X.shape[1]>1:
        for ip in range(1,p):
          y*= np.sin(2*np.pi*X[:,ip])
      return y.reshape((-1,1))
    
    x=3*np.random.standard_cauchy((n,1))
    y=f(x)+np.random.normal(0,noise,(n,p))
    x1=np.linspace(np.min(x),np.max(x),N).reshape((-1,1))
    SIGMA_BDS=[[1e-10,.4],[0.01,1],[0.01,1]]
    LBDAS=[0, 10, 30]
    Y_LIMS_A=[[None,1e3],[None,None],[None,None]]
    Y_LIMS_E=[[None,None],[None,None],[None,None]]
    sigma_maxs.append(2*n*np.exp(-1.5))
  
  axs=axs_m[r_axs,:]
  for ax,lbda,sigma_bds, y_lim_a, y_lim_e in zip(axs,LBDAS,SIGMA_BDS,Y_LIMS_A, Y_LIMS_E):
    sigmas=np.linspace(sigma_bds[0],sigma_bds[1],1000)
    if alg=='c':
      D=np.abs(x-x.T)
      sigma_j=opt_sigma_m(D,lbda)[0]
      j2a_0, j2a_1, j2a_2 = j2a_m(sigmas,x,lbda)
    else:
      sigma_j=opt_sigma(n,p,l,lbda)[0]
      j2a_0, j2a_1, j2a_2 = j2a(sigmas,n,p,l,lbda)
    if alg=='2d': sigma_j*=0.001
    
    ax1=ax.twinx()
    if alg=='2d':
      j2_emp_0= j2_emp_map(D,y,D1,lat_map,lon_map,lbda,sigmas)
      ax.plot(0.001*sigmas,j2a_0, 'C2',linewidth=2)
      ax1.plot(0.001*sigmas,j2_emp_0,'C0', linewidth=2)
    else:
      j2_emp_0= j2_emp(x,y,x1,lbda,sigmas)
      ax.plot(sigmas,j2a_0, 'C2',linewidth=2)
      ax1.plot(sigmas,j2_emp_0,'C0', linewidth=2)
    
    if lbda==0:
      ax.set_yscale('log')
      ax1.set_yscale('log')
      ax.set_ylabel(title, fontsize=FS_TITLE)
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

print(sigma_maxs)
fig.legend(lines, ['$J_2^a$','$J_2$','$\\sigma_0$'], loc='lower center', ncol=3, fontsize=FS_LEG)
plt.tight_layout()
fig.subplots_adjust(bottom=.09)
plt.savefig('figures/bw_sim_demo.pdf')

