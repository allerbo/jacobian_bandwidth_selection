import numpy as np
import pandas as pd
import sys
import time
sys.path.insert(1,'.')
from help_fcts import opt_sigma, r2, krr, get_sil, gcv, cv_10, opt_sigma_m, log_marg, log_marg_seed
from datetime import datetime as dt

temps_data = pd.read_csv('french_1d.csv', delimiter=";")

y_all = temps_data[['t']].values-273.15
x_temp = temps_data[['date']].values
x_temp1=list(map(lambda d: dt.strptime(str(d)[1:11],'%Y%m%d%H'),x_temp))
x_all=np.array(list(map(lambda d: (d-x_temp1[0]).total_seconds()/3600,x_temp1))).reshape((-1,1))

tp=sys.argv[1]

if tp=='n':
  lbda=1e-3
  n=int(sys.argv[2])
  suf='_n_'+str(n)
elif tp=='lbda':
  n=100
  log10lbda=int(sys.argv[2])
  lbda=10**(0.1*log10lbda)
  suf='_lbda_'+str(log10lbda)
else:
  print('Wrong type')
  sys.exit()

if len(sys.argv)>=4:
  for arg in range(3,len(sys.argv)):
    exec(sys.argv[arg])


p=1
N_SEEDS=1000

r2s_j=[]
r2s_jm=[]
r2s_cv=[]
r2s_gcv=[]
r2s_sil=[]
sigmas_j=[]
sigmas_jm=[]
sigmas_cv=[]
sigmas_gcv=[]
sigmas_sil=[]
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

for seed in range(N_SEEDS):
  np.random.seed(seed)
  per=np.random.permutation(len(x_all))
  if tp=='n':
    x, x_test=x_all[per[:n]], x_all[per[round(0.85*len(y_all)):]]
    y, y_test=y_all[per[:n]], y_all[per[round(0.85*len(y_all)):]]
  else:
    x, x_test=x_all[per[:n]], x_all[per[n:]]
    y, y_test=y_all[per[:n]], y_all[per[n:]]
  
  l=np.max(x)-np.min(x)
  
  #Jacobian
  t1=time.time()
  sigma_j=opt_sigma(n,p,l,lbda)[0]
  times_j.append(time.time()-t1)
  sigmas_j.append(sigma_j)

  #Jacobian M
  t1=time.time()
  sigma_jm=opt_sigma_m(x,lbda)[0]
  times_jm.append(time.time()-t1)
  sigmas_jm.append(sigma_jm)
  
  #Silverman
  t1=time.time()
  sigma_sil=get_sil(n,p,x)
  times_sil.append(time.time()-t1)
  sigmas_sil.append(sigma_sil)

  #GCV
  t1=time.time()
  sigmas=np.logspace(-3,np.log10(l),100)
  sigma_gcv=gcv(x,y,lbda,sigmas)
  times_gcv.append(time.time()-t1)
  sigmas_gcv.append(sigma_gcv)
  
  #CV
  t1=time.time()
  sigmas=np.logspace(-3,np.log10(l),100)
  sigma_cv=cv_10(x,y,lbda,sigmas,seed)
  times_cv.append(time.time()-t1)
  sigmas_cv.append(sigma_cv)

  #JCV
  t1=time.time()
  sigmas1=np.logspace(np.log10(sigma_j/3),np.log10(3*sigma_j),100)
  sigma_jcv=cv_10(x,y,lbda,sigmas1,seed)
  times_jcv.append(time.time()-t1)
  sigmas_jcv.append(sigma_jcv)

  #LM
  t1=time.time()
  sigma_lm=log_marg(x,y,lbda,(1e-3,l))
  times_lm.append(time.time()-t1)
  sigmas_lm.append(sigma_lm)

  #JLM
  t1=time.time()
  sigma_jlm=log_marg_seed(x,y,lbda,sigma_j,(1e-3,l))
  times_jlm.append(time.time()-t1)
  sigmas_jlm.append(sigma_jlm)


  y1_j=krr(x_test,x,y,sigma_j,lbda)
  y1_jm=krr(x_test,x,y,sigma_jm,lbda)
  y1_gcv=krr(x_test,x,y,sigma_gcv,lbda)
  y1_sil=krr(x_test,x,y,sigma_sil,lbda)
  r2s_j.append(r2(y_test,y1_j))
  r2s_jm.append(r2(y_test,y1_jm))
  r2s_gcv.append(r2(y_test,y1_gcv))
  r2s_sil.append(r2(y_test,y1_sil))
  y1_cv=krr(x_test,x,y,sigma_cv,lbda)
  r2s_cv.append(r2(y_test,y1_cv))
  y1_jcv=krr(x_test,x,y,sigma_jcv,lbda)
  r2s_jcv.append(r2(y_test,y1_jcv))
  y1_lm=krr(x_test,x,y,sigma_lm,lbda)
  r2s_lm.append(r2(y_test,y1_lm))
  y1_jlm=krr(x_test,x,y,sigma_jlm,lbda)
  r2s_jlm.append(r2(y_test,y1_jlm))

  krr(x_all,x_all,y_all,1,lbda) #To reset times
  
np.save('data_french_1d/french_1d_r2s_j'+suf,r2s_j)
np.save('data_french_1d/french_1d_r2s_jm'+suf,r2s_jm)
np.save('data_french_1d/french_1d_r2s_gcv'+suf,r2s_gcv)
np.save('data_french_1d/french_1d_r2s_sil'+suf,r2s_sil)
np.save('data_french_1d/french_1d_sigmas_j'+suf,sigmas_j)
np.save('data_french_1d/french_1d_sigmas_jm'+suf,sigmas_jm)
np.save('data_french_1d/french_1d_sigmas_gcv'+suf,sigmas_gcv)
np.save('data_french_1d/french_1d_sigmas_sil'+suf,sigmas_sil)
np.save('data_french_1d/french_1d_times_j'+suf,times_j)
np.save('data_french_1d/french_1d_times_jm'+suf,times_jm)
np.save('data_french_1d/french_1d_times_gcv'+suf,times_gcv)
np.save('data_french_1d/french_1d_times_sil'+suf,times_sil)

np.save('data_french_1d/french_1d_r2s_cv'+suf,r2s_cv)
np.save('data_french_1d/french_1d_sigmas_cv'+suf,sigmas_cv)
np.save('data_french_1d/french_1d_times_cv'+suf,times_cv)
np.save('data_french_1d/french_1d_r2s_jcv'+suf,r2s_jcv)
np.save('data_french_1d/french_1d_sigmas_jcv'+suf,sigmas_jcv)
np.save('data_french_1d/french_1d_times_jcv'+suf,times_jcv)
np.save('data_french_1d/french_1d_r2s_lm'+suf,r2s_lm)
np.save('data_french_1d/french_1d_sigmas_lm'+suf,sigmas_lm)
np.save('data_french_1d/french_1d_times_lm'+suf,times_lm)
np.save('data_french_1d/french_1d_r2s_jlm'+suf,r2s_jlm)
np.save('data_french_1d/french_1d_sigmas_jlm'+suf,sigmas_jlm)
np.save('data_french_1d/french_1d_times_jlm'+suf,times_jlm)

