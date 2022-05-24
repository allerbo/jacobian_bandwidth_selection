import numpy as np
import pandas as pd
from pymap3d import vincenty
import sys
sys.path.insert(1,'.')
from help_fcts import opt_sigma, r2, calc_D_map, krr_map, get_sil_std
from sklearn.model_selection import KFold



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
y1s_0=[]
y1s_cv=[]
y1s_sil=[]

i_del=int(sys.argv[1])
X=np.delete(X_all,i_del,0)
y=np.delete(y_all,i_del,0)

D=calc_D_map(X,X)
D1=calc_D_map(X1,X)

mean_dists=np.zeros(X.shape[0])
for i in range(X.shape[0]):
  mean_dists[i] = vincenty.vdist(X[i,0], X[i,1], np.mean(X[:,0]), np.mean(X[:,1]))[0]

n,p=X.shape
l=np.max(D)

#Jacobian
sigma0=opt_sigma(n,p,l,lbda)[0]

#Silverman
std_sil=np.sqrt(np.mean(mean_dists**2))
sigma_sil=get_sil_std(n,p,std_sil)

#CV
kf=KFold(n_splits=min(n,10),shuffle=True, random_state=i_del)
sigmas=np.logspace(1e-2,np.log10(l),100)
mse_mat=np.zeros((kf.n_splits,len(sigmas)))
for i, (ti,vi) in enumerate(kf.split(X,y)):
  Xt,Xv,yt,yv= X[ti,:], X[vi,:], y[ti,:], y[vi,:]
  D_cv=calc_D_map(Xt,Xt)
  D1_cv=calc_D_map(Xv,Xt)
  for j,sigma in enumerate(sigmas):
    y1=krr_map(D1_cv,D_cv,yt,sigma,lbda)
    mse_mat[i,j]=np.mean((yv-y1)**2)

mses_mean=np.mean(mse_mat,0)
sigma_cv=sigmas[np.argmin(mses_mean)]

y1_0=krr_map(D1,D,y,sigma0,lbda)
y1_sil=krr_map(D1,D,y,sigma_sil,lbda)
y1_cv=krr_map(D1,D,y,sigma_cv,lbda)

np.save('data/french_2d_y1_0_jk_'+str(i_del),y1_0)
np.save('data/french_2d_y1_cv_jk_'+str(i_del),y1_cv)
np.save('data/french_2d_y1_sil_jk_'+str(i_del),y1_sil)
np.save('data/french_2d_sigma_0_jk_'+str(i_del),sigma0)
np.save('data/french_2d_sigma_cv_jk_'+str(i_del),sigma_cv)
np.save('data/french_2d_sigma_sil_jk_'+str(i_del),sigma_sil)

