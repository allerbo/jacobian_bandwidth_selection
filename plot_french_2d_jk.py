import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
sys.path.insert(1,'.')
from help_fcts import std_jk
from matplotlib import pyplot as plt
import glob

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



del_strs=glob.glob('data/french_2d_y1_0_jk_*npy')
i_dels=sorted(list(map(lambda s: int(s.split('_')[-1][:-4]), del_strs)))
  
y1s_0=[]
y1s_cv=[]
y1s_sil=[]
sigmas_0=[]
sigmas_cv=[]
sigmas_sil=[]

for i_del in i_dels:
  y1s_0.append(np.load('data/french_2d_y1_0_jk_'+str(i_del)+'.npy'))
  y1s_cv.append(np.load('data/french_2d_y1_cv_jk_'+str(i_del)+'.npy'))
  y1s_sil.append(np.load('data/french_2d_y1_sil_jk_'+str(i_del)+'.npy'))
  sigmas_0.append(np.load('data/french_2d_sigma_0_jk_'+str(i_del)+'.npy'))
  sigmas_cv.append(np.load('data/french_2d_sigma_cv_jk_'+str(i_del)+'.npy'))
  sigmas_sil.append(np.load('data/french_2d_sigma_sil_jk_'+str(i_del)+'.npy'))

def scale_mean(x, a=2, b=0.1):
  return np.tanh(b*(x-a))

def scale_std(x, a=0, b=0.1):
  return np.tanh(b*(x-a))


fig,axs=plt.subplots(2,3,figsize=(11,7), subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={'width_ratios': [1,1,1.23]})

ticks_mean=np.array([-5.,  0.,  5., 10., 15., 20.])
ticks_std=np.array([ 0.,  5., 10., 15., 20.])

map_type = "pcolor" 
for ii, (agg_fct, scale_agg_fct, ticks) in enumerate(zip([np.mean, std_jk],[scale_mean,scale_std], [ticks_mean, ticks_std])):
  y_max=np.max(np.hstack((agg_fct(np.hstack(y1s_0),1), agg_fct(np.hstack(y1s_cv),1), agg_fct(np.hstack(y1s_sil),1))))
  y_min=np.min(np.hstack((agg_fct(np.hstack(y1s_0),1), agg_fct(np.hstack(y1s_cv),1), agg_fct(np.hstack(y1s_sil),1))))
  if ii==1:
    y_min_s=0
  else:
    y_min_s=scale_agg_fct(y_min)
  y_max_s=scale_agg_fct(y_max)
  for jj, (y1s, ax, title,sigmas) in enumerate(zip([y1s_0,y1s_cv,y1s_sil],axs[ii,:], ['Jacobian', 'Cross-validation', 'Silverman'],[sigmas_0,sigmas_cv,sigmas_sil])):
    y1=agg_fct(np.hstack(y1s),1)
    y1_s=scale_agg_fct(y1)
    map_y1=y1_s.reshape((n_width, n_width))
    
    im = ax.pcolormesh(lon_map_uni, lat_map_uni, (map_y1).T, vmin=y_min_s, vmax=y_max_s, cmap="jet")
    if ii==0:
      ax.scatter(lons_st, lats_st, c=scale_agg_fct(y_all), cmap='jet', s=80, edgecolor="black", vmin=im.get_clim()[0], vmax=im.get_clim()[1])
    if jj==2:
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



