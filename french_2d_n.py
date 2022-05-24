import numpy as np
import pandas as pd
import sys
sys.path.insert(1,'.')
from pymap3d import vincenty
from help_fcts import opt_sigma, krr_map, calc_D_map, r2, get_sil_std
from sklearn.model_selection import KFold


fpd = pd.read_csv('french_2d.csv', delimiter=";")
lats_st = fpd.Latitude.to_numpy()
lons_st = fpd.Longitude.to_numpy()
X_all=np.vstack((lats_st,lons_st)).T

y_all = fpd[['t']].values-273.15

n=int(sys.argv[1])
r2s_0=[]
r2s_sil=[]
r2s_cv=[]
sigmas_0=[]
sigmas_sil=[]
sigmas_cv=[]
r2s_cv1=[]
sigmas_cv1=[]

lbda=1e-3

N_SEEDS=1000

for seed in range(N_SEEDS):
  np.random.seed(seed)
  per=np.random.permutation(X_all.shape[0])
  X, X_test=X_all[per[:n],:], X_all[per[round(0.85*len(y_all)):],:]
  y, y_test=y_all[per[:n]], y_all[per[round(0.85*len(y_all)):]]
  
  D=calc_D_map(X,X)
  D1=calc_D_map(X_test,X)
  
  mean_dists=np.zeros(X.shape[0])
  for i in range(X.shape[0]):
    mean_dists[i] = vincenty.vdist(X[i,0], X[i,1], np.mean(X[:,0]), np.mean(X[:,1]))[0]
  
  n,p=X.shape
  l=np.max(D)
  
  #Jacobian
  sigma0=opt_sigma(n,p,l,lbda)[0]
  sigmas_0.append(sigma0)
  
  #Silverman
  std_sil=np.sqrt(np.mean(mean_dists**2))
  sigma_sil=get_sil_std(n,p,std_sil)
  sigmas_sil.append(sigma_sil)
  
  #CV
  kf=KFold(n_splits=min(n,10),shuffle=True, random_state=seed)
  sigmas=np.logspace(2,np.log10(l),100)
  
  mse_mat=np.zeros((kf.n_splits,len(sigmas)))
  for i, (ti,vi) in enumerate(kf.split(X,y)):
    Xt,Xv,yt,yv= X[ti,:], X[vi,:], y[ti], y[vi]
    D_cv=calc_D_map(Xt,Xt)
    D1_cv=calc_D_map(Xv,Xt)
    for j,sigma in enumerate(sigmas):
      y1=krr_map(D1_cv,D_cv,yt-np.mean(yt),sigma,lbda)+np.mean(yt)
      mse_mat[i,j]=np.mean((yv-y1)**2)
  
  mses_mean=np.mean(mse_mat,0)
  sigma_cv=sigmas[np.argmin(mses_mean)]
  sigmas_cv.append(sigma_cv)
  
  #Jacobian seeded CV
  kf=KFold(n_splits=min(n,10),shuffle=True, random_state=seed)
  sigmas=np.logspace(np.log10(0.2*sigma0),np.log10(5*sigma0),100)
  
  mse_mat=np.zeros((kf.n_splits,len(sigmas)))
  for i, (ti,vi) in enumerate(kf.split(X,y)):
    Xt,Xv,yt,yv= X[ti,:], X[vi,:], y[ti], y[vi]
    D_cv=calc_D_map(Xt,Xt)
    D1_cv=calc_D_map(Xv,Xt)
    for j,sigma in enumerate(sigmas):
      y1=krr_map(D1_cv,D_cv,yt-np.mean(yt),sigma,lbda)+np.mean(yt)
      mse_mat[i,j]=np.mean((yv-y1)**2)
  
  mses_mean=np.mean(mse_mat,0)
  sigma_cv1=sigmas[np.argmin(mses_mean)]
  sigmas_cv1.append(sigma_cv1)
  
  y1_0=krr_map(D1,D,y-np.mean(y),sigma0,lbda)+np.mean(y)
  y1_sil=krr_map(D1,D,y-np.mean(y),sigma_sil,lbda)+np.mean(y)
  y1_cv=krr_map(D1,D,y-np.mean(y),sigma_cv,lbda)+np.mean(y)
  
  r2s_0.append(r2(y_test,y1_0))
  r2s_sil.append(r2(y_test,y1_sil))
  r2s_cv.append(r2(y_test,y1_cv))
  
  y1_cv1=krr_map(D1,D,y-np.mean(y),sigma_cv1,lbda)+np.mean(y)
  r2s_cv1.append(r2(y_test,y1_cv1))

np.save('data/french_2d_r2s_0_n_'+str(n),r2s_0)
np.save('data/french_2d_r2s_cv_n_'+str(n),r2s_cv)
np.save('data/french_2d_r2s_sil_n_'+str(n),r2s_sil)
np.save('data/french_2d_sigmas_0_n_'+str(n),sigmas_0)
np.save('data/french_2d_sigmas_cv_n_'+str(n),sigmas_cv)
np.save('data/french_2d_sigmas_sil_n_'+str(n),sigmas_sil)

np.save('data/french_2d_r2s_cv1_n_'+str(n),r2s_cv1)
np.save('data/french_2d_sigmas_cv1_n_'+str(n),sigmas_cv1)

