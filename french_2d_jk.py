import numpy as np
import pandas as pd
from pymap3d import vincenty
import sys
sys.path.insert(1,'.')
from help_fcts import opt_sigma, r2, calc_D_map, krr_map, get_sil_std, gcv_map, cv_10_map, log_marg_map, log_marg_map_seed, std_jk
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import pyplot as plt


fpd = pd.read_csv('french_2d.csv', delimiter=";")
lats_st = fpd.Latitude.to_numpy()
lons_st = fpd.Longitude.to_numpy()
X_all=np.vstack((lats_st,lons_st)).T

y_all = fpd[['t']].values-273.15

bound_map = ((42.3, -5), (51.1, 8.3))

# Create points for interpolation
n_width = 100 
lat_map_uni = np.linspace(bound_map[0][0], bound_map[1][0], n_width)
lon_map_uni = np.linspace(bound_map[0][1], bound_map[1][1], n_width)
lat_map, lon_map = np.meshgrid(lat_map_uni, lon_map_uni)
lat_map_ = lat_map.ravel()
lon_map_ = lon_map.ravel()
X1=np.vstack((lat_map_,lon_map_)).T

lbda=1e-3
sigmas_j=[]
sigmas_gcv=[]
sigmas_sil=[]
sigmas_lm=[]
y1s_j=[]
y1s_gcv=[]
y1s_sil=[]
y1s_lm=[]

D_all=calc_D_map(X_all,X_all)
D1_all=calc_D_map(X1,X_all)

for i_del in range(40):
  print(i_del)
  D=np.delete(np.delete(D_all,i_del,0),i_del,1)
  D1=np.delete(D1_all,i_del,1)
  X=np.delete(X_all,i_del,0)
  y=np.delete(y_all,i_del,0)
  
  
  n,p=X.shape
  n=D.shape[0]
  l=np.max(D)
  
  #Jacobian
  sigma_j=opt_sigma(n,p,l,lbda)[0]
  sigmas_j.append(sigma_j)
  
  #Silverman
  mean_dists=np.zeros(X.shape[0])
  for i in range(X.shape[0]):
    mean_dists[i] = vincenty.vdist(X[i,0], X[i,1], np.mean(X[:,0]), np.mean(X[:,1]))[0]
  
  std_sil=np.sqrt(np.mean(mean_dists**2))
  sigma_sil=get_sil_std(n,p,std_sil)
  sigmas_sil.append(sigma_sil)
  
  #GCV
  sigmas=np.logspace(-3,np.log10(l),100)
  sigma_gcv=gcv_map(D,y,lbda,sigmas)
  sigmas_gcv.append(sigma_gcv)
  
  #LM
  sigma_lm=log_marg_map(D,y,lbda,(1e-3,l))
  sigmas_lm.append(sigma_lm)
  
  y1_j=krr_map(D1,D,y,sigma_j,lbda)
  y1_sil=krr_map(D1,D,y,sigma_sil,lbda)
  y1_gcv=krr_map(D1,D,y,sigma_gcv,lbda)
  y1_lm=krr_map(D1,D,y,sigma_lm,lbda)
  
  y1s_j.append(y1_j)
  y1s_gcv.append(y1_gcv)
  y1s_sil.append(y1_sil)
  y1s_lm.append(y1_lm)


def scale_mean(x, a=2, b=0.1):
  return np.tanh(b*(x-a))

def scale_std(x, a=0, b=0.1):
  return np.tanh(b*(x-a))


fig,axs=plt.subplots(2,4,figsize=(0.95*12,0.95*6), subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={'width_ratios': [1,1,1,1.23]})

ticks_mean=np.array([-5.,  0.,  5., 10., 15., 20.])
ticks_std=np.array([ 0.,  5., 10., 15., 20.])

map_type = "pcolor" 
for ii, (agg_fct, scale_agg_fct, ticks) in enumerate(zip([np.mean, std_jk],[scale_mean,scale_std], [ticks_mean, ticks_std])):
  y_max=np.max(np.hstack((agg_fct(np.hstack(y1s_j),1), agg_fct(np.hstack(y1s_gcv),1), agg_fct(np.hstack(y1s_sil),1))))
  y_min=np.min(np.hstack((agg_fct(np.hstack(y1s_j),1), agg_fct(np.hstack(y1s_gcv),1), agg_fct(np.hstack(y1s_sil),1))))
  if ii==1:
    y_min_s=0
  else:
    y_min_s=scale_agg_fct(y_min)
  y_max_s=scale_agg_fct(y_max)
  for jj, (y1s, ax, title,sigmas) in enumerate(zip([y1s_j,y1s_gcv,y1s_lm,y1s_sil],axs[ii,:], ['Jacobian', 'GCV', 'MML', 'Silverman'],[sigmas_j,sigmas_gcv,sigmas_lm,sigmas_sil])):
    y1=agg_fct(np.hstack(y1s),1)
    y1_s=scale_agg_fct(y1)
    map_y1=y1_s.reshape((n_width, n_width))
    
    im = ax.pcolormesh(lon_map_uni, lat_map_uni, (map_y1).T, vmin=y_min_s, vmax=y_max_s, cmap="jet")
    if ii==0:
      ax.scatter(lons_st, lats_st, c=scale_agg_fct(y_all), cmap='jet', s=80, edgecolor="black", vmin=im.get_clim()[0], vmax=im.get_clim()[1])
    if jj==3:
      cb=plt.colorbar(im,ax=ax)
      cb.set_ticks(scale_agg_fct(ticks))
      cb.set_ticklabels(ticks)
    ax.margins(0)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
    ax.coastlines()
    if ii==0:
      ax.set_title(title+'. Mean$(\\sigma)=$'+str(round(np.mean(sigmas)/1000))+' km', fontsize=12.8)
    elif ii==1:
      ax.set_title(title+'. Std$(\\sigma)=$'+str(round(std_jk(sigmas)/1000))+' km', fontsize=12.8)
    
    ax.set_aspect("auto")
    

plt.tight_layout()
plt.savefig('figures/french_2d_jk.pdf')



