import numpy as np
import pandas as pd
from pymap3d import vincenty
import sys
sys.path.insert(1,'.')
from help_fcts import opt_sigma, r2, calc_D_map, krr_map, get_sil_std, gcv_map, cv_10_map, log_marg_map, log_marg_map_seed



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
y1s_j=[]
y1s_cv=[]
y1s_sil=[]
y1s_lm=[]
y1s_jlm=[]

i_del=int(sys.argv[1])
X=np.delete(X_all,i_del,0)
y=np.delete(y_all,i_del,0)

D=calc_D_map(X,X)
D1=calc_D_map(X1,X)


n,p=X.shape
l=np.max(D)

#Jacobian
sigma_j=opt_sigma(n,p,l,lbda)[0]

#Silverman
mean_dists=np.zeros(X.shape[0])
for i in range(X.shape[0]):
  mean_dists[i] = vincenty.vdist(X[i,0], X[i,1], np.mean(X[:,0]), np.mean(X[:,1]))[0]

std_sil=np.sqrt(np.mean(mean_dists**2))
sigma_sil=get_sil_std(n,p,std_sil)

#GCV
sigmas=np.logspace(-3,np.log10(l),100)
sigma_gcv=gcv_map(D,y,lbda,sigmas)

#CV
sigmas=np.logspace(-3,np.log10(l),100)
sigma_cv=cv_10_map(D,y,lbda,sigmas,i_del)

#LM
sigma_lm=log_marg_map(D,y,lbda,(1e-3,l))

#JLM
sigma_jlm=log_marg_map_seed(D,y,lbda,sigma_j,(1e-3,l))

y1_j=krr_map(D1,D,y,sigma_j,lbda)
y1_sil=krr_map(D1,D,y,sigma_sil,lbda)
y1_gcv=krr_map(D1,D,y,sigma_gcv,lbda)
y1_cv=krr_map(D1,D,y,sigma_cv,lbda)
y1_lm=krr_map(D1,D,y,sigma_lm,lbda)
y1_jlm=krr_map(D1,D,y,sigma_jlm,lbda)

np.save('data_french_2d/french_2d_y1_j_jk_'+str(i_del),y1_j)
np.save('data_french_2d/french_2d_y1_cv_jk_'+str(i_del),y1_cv)
np.save('data_french_2d/french_2d_y1_gcv_jk_'+str(i_del),y1_gcv)
np.save('data_french_2d/french_2d_y1_sil_jk_'+str(i_del),y1_sil)
np.save('data_french_2d/french_2d_sigma_j_jk_'+str(i_del),sigma_j)
np.save('data_french_2d/french_2d_sigma_cv_jk_'+str(i_del),sigma_cv)
np.save('data_french_2d/french_2d_sigma_gcv_jk_'+str(i_del),sigma_gcv)
np.save('data_french_2d/french_2d_sigma_sil_jk_'+str(i_del),sigma_sil)

np.save('data_french_2d/french_2d_y1_lm_jk_'+str(i_del),y1_lm)
np.save('data_french_2d/french_2d_sigma_lm_jk_'+str(i_del),sigma_lm)
np.save('data_french_2d/french_2d_y1_jlm_jk_'+str(i_del),y1_jlm)
np.save('data_french_2d/french_2d_sigma_jlm_jk_'+str(i_del),sigma_jlm)
