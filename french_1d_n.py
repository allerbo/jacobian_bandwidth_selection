import numpy as np
import pandas as pd
import sys
sys.path.insert(1,'.')
from sklearn.model_selection import  KFold
from help_fcts import opt_sigma, r2, krr, get_sil
from datetime import datetime as dt

temps_data = pd.read_csv('french_1d.csv', delimiter=";")

y_all = temps_data[['t']].values-273.15
x_temp = temps_data[['date']].values
x_temp1=list(map(lambda d: dt.strptime(str(d)[1:11],'%Y%m%d%H'),x_temp))
x_all=np.array(list(map(lambda d: (d-x_temp1[0]).total_seconds()/3600,x_temp1))).reshape((-1,1))

p=1
lbda=1e-3
N_SEEDS=1000

n=int(sys.argv[1])
r2s_0=[]
r2s_cv=[]
r2s_sil=[]
sigmas_0=[]
sigmas_cv=[]
sigmas_sil=[]
r2s_cv1=[]
sigmas_cv1=[]
for seed in range(N_SEEDS):
  np.random.seed(seed)
  per=np.random.permutation(len(x_all))
  x, x_test=x_all[per[:n]], x_all[per[round(0.85*len(y_all)):]]
  y, y_test=y_all[per[:n]], y_all[per[round(0.85*len(y_all)):]]
  l=np.max(x)-np.min(x)
  
  #Jacobian
  sigma0=opt_sigma(n,p,l,lbda)[0]
  sigmas_0.append(sigma0)
  
  #Silverman
  sigma_sil=get_sil(n,p,x)
  sigmas_sil.append(sigma_sil)
  
  #CV
  kf=KFold(n_splits=min(n,10),shuffle=True, random_state=seed)
  sigmas=np.logspace(-2,np.log10(l),100)
  mses_o=[]
  for sigma in sigmas:
    mses=[]
    for ti,vi in kf.split(x,y):
      xt,xv,yt,yv= x[ti], x[vi], y[ti], y[vi,:]
      y1=krr(xv,xt,yt-np.mean(yt),sigma,lbda)+np.mean(yt)
      mses.append(np.mean((yv-y1)**2))
    mses_o.append(mses)
  
  mses_mean=np.mean(np.array(mses_o),1)
  sigma_cv=sigmas[np.argmin(mses_mean)]
  sigmas_cv.append(sigma_cv)
  
  #Jacobian seeded CV
  kf=KFold(n_splits=min(n,10),shuffle=True, random_state=seed)
  sigmas=np.logspace(np.log10(0.2*sigma0),np.log10(5*sigma0),100)
  mses_o=[]
  for sigma in sigmas:
    mses=[]
    for ti,vi in kf.split(x,y):
      xt,xv,yt,yv= x[ti], x[vi], y[ti], y[vi,:]
      y1=krr(xv,xt,yt-np.mean(yt),sigma,lbda)+np.mean(yt)
      mses.append(np.mean((yv-y1)**2))
    mses_o.append(mses)
  
  mses_mean=np.mean(np.array(mses_o),1)
  sigma_cv1=sigmas[np.argmin(mses_mean)]
  sigmas_cv1.append(sigma_cv1)
  
  y1_0=krr(x_test,x,y,sigma0,lbda)
  y1_cv=krr(x_test,x,y,sigma_cv,lbda)
  y1_sil=krr(x_test,x,y,sigma_sil,lbda)
  r2s_0.append(r2(y_test,y1_0))
  r2s_cv.append(r2(y_test,y1_cv))
  r2s_sil.append(r2(y_test,y1_sil))
  y1_cv1=krr(x_test,x,y,sigma_cv1,lbda)
  r2s_cv1.append(r2(y_test,y1_cv1))
  
np.save('data/french_1d_r2s_0_n_'+str(n),r2s_0)
np.save('data/french_1d_r2s_cv_n_'+str(n),r2s_cv)
np.save('data/french_1d_r2s_sil_n_'+str(n),r2s_sil)
np.save('data/french_1d_sigmas_0_n_'+str(n),sigmas_0)
np.save('data/french_1d_sigmas_cv_n_'+str(n),sigmas_cv)
np.save('data/french_1d_sigmas_sil_n_'+str(n),sigmas_sil)
np.save('data/french_1d_r2s_cv1_n_'+str(n),r2s_cv1)
np.save('data/french_1d_sigmas_cv1_n_'+str(n),sigmas_cv1)
