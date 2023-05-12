import numpy as np
import pandas as pd
import sys
import time
import pickle
sys.path.insert(1,'.')
from pymap3d import vincenty
from help_fcts import opt_sigma, krr_map, calc_D_map, r2, get_sil_std, gcv_map, cv_10_map, opt_sigma_m_map, log_marg_map, log_marg_map_seed


fpd = pd.read_csv('french_2d.csv', delimiter=";")
lats_st = fpd.Latitude.to_numpy()
lons_st = fpd.Longitude.to_numpy()
X_all=np.vstack((lats_st,lons_st)).T

y_all = fpd[['t']].values-273.15

n_sigmas=100
lbda=1e-3
seed=int(sys.argv[1])
suf='_n_'+str(seed)

if len(sys.argv)>=3:
  for arg in range(2,len(sys.argv)):
    exec(sys.argv[arg])


out_dict={}
for alg in ['j','gcv','lm','sil','jcv','jlm']:
  out_dict[alg]={}
  for metric in ['r2','sigma','time']:
    out_dict[alg][metric]=[]

for n in range(37,9,-1):
  np.random.seed(seed)
  per=np.random.permutation(X_all.shape[0])
  X, X_test=X_all[per[:n],:], X_all[per[round(0.85*len(y_all)):],:]
  y, y_test=y_all[per[:n]], y_all[per[round(0.85*len(y_all)):]]
  
  D=calc_D_map(X,X)
  n,p=X.shape
  l=np.max(D)

  #Jacobian
  t1=time.time()
  sigma_j=opt_sigma(n,p,l,lbda)[0]
  out_dict['j']['time'].append(time.time()-t1)
  out_dict['j']['sigma'].append(sigma_j)
  
  
  #Silverman
  t1=time.time()
  n,p=X.shape
  mean_dists=np.zeros(X.shape[0])
  for i in range(X.shape[0]):
    mean_dists[i] = vincenty.vdist(X[i,0], X[i,1], np.mean(X[:,0]), np.mean(X[:,1]))[0]
  std_sil=np.sqrt(np.mean(mean_dists**2))
  sigma_sil=get_sil_std(n,p,std_sil)
  out_dict['sil']['time'].append(time.time()-t1)
  out_dict['sil']['sigma'].append(sigma_sil)

  #GCV
  t1=time.time()
  sigmas=np.logspace(-3,np.log10(l),n_sigmas)
  sigma_gcv=gcv_map(D,y,lbda,sigmas)
  out_dict['gcv']['time'].append(time.time()-t1)
  out_dict['gcv']['sigma'].append(sigma_gcv)

  #LM
  t1=time.time()
  sigma_lm=log_marg_map(D,y,lbda,(1e-3,l))
  out_dict['lm']['time'].append(time.time()-t1)
  out_dict['lm']['sigma'].append(sigma_lm)

  #JCV
  t1=time.time()
  sigmas1=np.logspace(np.log10(sigma_j/3),np.log10(3*sigma_j),n_sigmas)
  sigma_jcv=gcv_map(D,y,lbda,sigmas1)
  out_dict['jcv']['time'].append(time.time()-t1)
  out_dict['jcv']['sigma'].append(sigma_jcv)
  
  #JLM
  t1=time.time()
  sigma_jlm=log_marg_map_seed(D,y,lbda,sigma_j)
  out_dict['jlm']['time'].append(time.time()-t1)
  out_dict['jlm']['sigma'].append(sigma_jlm)


  D1=calc_D_map(X_test,X)
  
  y1_j=krr_map(D1,D,y,sigma_j,lbda)
  y1_gcv=krr_map(D1,D,y,sigma_gcv,lbda)
  y1_lm=krr_map(D1,D,y,sigma_lm,lbda)
  y1_sil=krr_map(D1,D,y,sigma_sil,lbda)
  y1_jcv=krr_map(D1,D,y,sigma_jcv,lbda)
  y1_jlm=krr_map(D1,D,y,sigma_jlm,lbda)

  out_dict['j']['r2'].append(r2(y_test,y1_j))
  out_dict['gcv']['r2'].append(r2(y_test,y1_gcv))
  out_dict['lm']['r2'].append(r2(y_test,y1_lm))
  out_dict['sil']['r2'].append(r2(y_test,y1_sil))
  out_dict['jcv']['r2'].append(r2(y_test,y1_jcv))
  out_dict['jlm']['r2'].append(r2(y_test,y1_jlm))

  D_all=calc_D_map(X_all,X_all)
  krr_map(D_all,D_all,y_all,1,lbda) #To reset times

fi=open('data1/french_2d'+suf+'.pkl','wb')
pickle.dump(out_dict,fi)
fi.close()
