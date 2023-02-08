import numpy as np
import pandas as pd
import sys
import time
sys.path.insert(1,'.')
from pymap3d import vincenty
from help_fcts import opt_sigma, krr_map, calc_D_map, r2, get_sil_std, gcv_map, cv_10_map, opt_sigma_m_map, log_marg_map, log_marg_map_seed


fpd = pd.read_csv('french_2d.csv', delimiter=";")
lats_st = fpd.Latitude.to_numpy()
lons_st = fpd.Longitude.to_numpy()
X_all=np.vstack((lats_st,lons_st)).T

y_all = fpd[['t']].values-273.15

tp=sys.argv[1]

if tp=='n':
  lbda=1e-3
  n=int(sys.argv[2])
  suf='_n_'+str(n)
elif tp=='lbda':
  n=25
  log10lbda=int(sys.argv[2])
  lbda=10**(0.1*log10lbda)
  suf='_lbda_'+str(log10lbda)
else:
  print('Wrong type')
  sys.exit()

if len(sys.argv)>=4:
  for arg in range(3,len(sys.argv)):
    exec(sys.argv[arg])


r2s_j=[]
r2s_jm=[]
r2s_sil=[]
r2s_cv=[]
r2s_gcv=[]
sigmas_j=[]
sigmas_jm=[]
sigmas_sil=[]
sigmas_cv=[]
sigmas_gcv=[]
times_j=[]
times_jm=[]
times_cv=[]
times_gcv=[]
times_sil=[]
r2s_jcv=[]
sigmas_jcv=[]
times_jcv=[]
r2s_lm=[]
sigmas_lm=[]
times_lm=[]
r2s_jlm=[]
sigmas_jlm=[]
times_jlm=[]




N_SEEDS=1000

for seed in range(N_SEEDS):
  np.random.seed(seed)
  per=np.random.permutation(X_all.shape[0])
  if tp=='n':
    X, X_test=X_all[per[:n],:], X_all[per[round(0.85*len(y_all)):],:]
    y, y_test=y_all[per[:n]], y_all[per[round(0.85*len(y_all)):]]
  else:
    X, X_test=X_all[per[:n],:], X_all[per[n:],:]
    y, y_test=y_all[per[:n]], y_all[per[n:]]
  
  
  D=calc_D_map(X,X)
  n,p=X.shape
  l=np.max(D)

  #Jacobian
  t1=time.time()
  sigma_j=opt_sigma(n,p,l,lbda)[0]
  sigmas_j.append(sigma_j)
  times_j.append(time.time()-t1)
  
  #Jacobian M
  t1=time.time()
  sigma_jm=opt_sigma_m_map(D,lbda)[0]
  sigmas_jm.append(sigma_jm)
  times_jm.append(time.time()-t1)
  
  
  #Silverman
  t1=time.time()
  n,p=X.shape
  mean_dists=np.zeros(X.shape[0])
  for i in range(X.shape[0]):
    mean_dists[i] = vincenty.vdist(X[i,0], X[i,1], np.mean(X[:,0]), np.mean(X[:,1]))[0]
  std_sil=np.sqrt(np.mean(mean_dists**2))
  sigma_sil=get_sil_std(n,p,std_sil)
  sigmas_sil.append(sigma_sil)
  times_sil.append(time.time()-t1)

  #CV
  t1=time.time()
  sigmas=np.logspace(-3,np.log10(l),100)
  sigma_cv=cv_10_map(D,y,lbda,sigmas,seed)
  sigmas_cv.append(sigma_cv)
  times_cv.append(time.time()-t1)
  

  #GCV
  t1=time.time()
  sigmas=np.logspace(-3,np.log10(l),100)
  sigma_gcv=gcv_map(D,y,lbda,sigmas)
  sigmas_gcv.append(sigma_gcv)
  times_gcv.append(time.time()-t1)

  #JCV
  t1=time.time()
  sigmas1=np.logspace(np.log10(sigma_j/3),np.log10(3*sigma_j),100)
  sigma_jcv=cv_10_map(D,y,lbda,sigmas1,seed)
  sigmas_jcv.append(sigma_jcv)
  times_jcv.append(time.time()-t1)
  
  #LM
  t1=time.time()
  sigma_lm=log_marg_map(D,y,lbda,(1e-3,l))
  times_lm.append(time.time()-t1)
  sigmas_lm.append(sigma_lm)

  #JLM
  t1=time.time()
  sigma_jlm=log_marg_map_seed(D,y,lbda,sigma_j,(1e-3,l))
  times_jlm.append(time.time()-t1)
  sigmas_jlm.append(sigma_jlm)


  D1=calc_D_map(X_test,X)
  
  y1_j=krr_map(D1,D,y,sigma_j,lbda)
  y1_jm=krr_map(D1,D,y,sigma_jm,lbda)
  y1_gcv=krr_map(D1,D,y,sigma_gcv,lbda)
  y1_sil=krr_map(D1,D,y,sigma_sil,lbda)
  
  r2s_j.append(r2(y_test,y1_j))
  r2s_jm.append(r2(y_test,y1_jm))
  r2s_gcv.append(r2(y_test,y1_gcv))
  r2s_sil.append(r2(y_test,y1_sil))

  y1_cv=krr_map(D1,D,y,sigma_cv,lbda)
  r2s_cv.append(r2(y_test,y1_cv))
  y1_jcv=krr_map(D1,D,y,sigma_jcv,lbda)
  r2s_jcv.append(r2(y_test,y1_jcv))
  y1_lm=krr_map(D1,D,y,sigma_lm,lbda)
  r2s_lm.append(r2(y_test,y1_lm))
  y1_jlm=krr_map(D1,D,y,sigma_jlm,lbda)
  r2s_jlm.append(r2(y_test,y1_jlm))


  D_all=calc_D_map(X_all,X_all)
  krr_map(D_all,D_all,y_all,1,lbda) #To reset times

np.save('data_french_2d/french_2d_r2s_j'+suf,r2s_j)
np.save('data_french_2d/french_2d_r2s_jm'+suf,r2s_jm)
np.save('data_french_2d/french_2d_r2s_gcv'+suf,r2s_gcv)
np.save('data_french_2d/french_2d_r2s_sil'+suf,r2s_sil)
np.save('data_french_2d/french_2d_sigmas_j'+suf,sigmas_j)
np.save('data_french_2d/french_2d_sigmas_jm'+suf,sigmas_jm)
np.save('data_french_2d/french_2d_sigmas_gcv'+suf,sigmas_gcv)
np.save('data_french_2d/french_2d_sigmas_sil'+suf,sigmas_sil)
np.save('data_french_2d/french_2d_times_j'+suf,times_j)
np.save('data_french_2d/french_2d_times_jm'+suf,times_jm)
np.save('data_french_2d/french_2d_times_gcv'+suf,times_gcv)
np.save('data_french_2d/french_2d_times_sil'+suf,times_sil)
np.save('data_french_2d/french_2d_r2s_cv'+suf,r2s_cv)
np.save('data_french_2d/french_2d_sigmas_cv'+suf,sigmas_cv)
np.save('data_french_2d/french_2d_times_cv'+suf,times_cv)
np.save('data_french_2d/french_2d_r2s_jcv'+suf,r2s_jcv)
np.save('data_french_2d/french_2d_sigmas_jcv'+suf,sigmas_jcv)
np.save('data_french_2d/french_2d_times_jcv'+suf,times_jcv)
np.save('data_french_2d/french_2d_r2s_lm'+suf,r2s_lm)
np.save('data_french_2d/french_2d_sigmas_lm'+suf,sigmas_lm)
np.save('data_french_2d/french_2d_times_lm'+suf,times_lm)
np.save('data_french_2d/french_2d_r2s_jlm'+suf,r2s_jlm)
np.save('data_french_2d/french_2d_sigmas_jlm'+suf,sigmas_jlm)
np.save('data_french_2d/french_2d_times_jlm'+suf,times_jlm)
