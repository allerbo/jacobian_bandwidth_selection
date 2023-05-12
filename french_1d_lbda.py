import numpy as np
import pandas as pd
import sys
import time
import pickle
sys.path.insert(1,'.')
from help_fcts import opt_sigma, r2, krr, get_sil, gcv, cv_10, opt_sigma_m, log_marg, log_marg_seed
from datetime import datetime as dt

temps_data = pd.read_csv('french_1d.csv', delimiter=";")

y_all = temps_data[['t']].values-273.15
x_temp = temps_data[['date']].values
x_temp1=list(map(lambda d: dt.strptime(str(d)[1:11],'%Y%m%d%H'),x_temp))
x_all=np.array(list(map(lambda d: (d-x_temp1[0]).total_seconds()/3600,x_temp1))).reshape((-1,1))


n_sigmas=100
seed=int(sys.argv[1])
n=100
suf='_lbda_'+str(seed)

if len(sys.argv)>=3:
  for arg in range(2,len(sys.argv)):
    exec(sys.argv[arg])


p=1


out_dict={}
for alg in ['j','gcv','lm','sil']:#,'jcv','jlm']:
  out_dict[alg]={}
  for metric in ['r2','sigma','time']:
    out_dict[alg][metric]=[]

for lbda in np.logspace(-5,2,15):
  np.random.seed(seed)
  per=np.random.permutation(len(x_all))
  x, x_test=x_all[per[:n]], x_all[per[n:]]
  y, y_test=y_all[per[:n]], y_all[per[n:]]
  
  l=np.max(x)-np.min(x)
  
  #Jacobian
  t1=time.time()
  sigma_j=opt_sigma(n,p,l,lbda)[0]
  out_dict['j']['time'].append(time.time()-t1)
  out_dict['j']['sigma'].append(sigma_j)

  #Silverman
  t1=time.time()
  sigma_sil=get_sil(n,p,x)
  out_dict['sil']['time'].append(time.time()-t1)
  out_dict['sil']['sigma'].append(sigma_sil)

  #GCV
  t1=time.time()
  sigmas=np.logspace(-3,np.log10(l),n_sigmas)
  sigma_gcv=gcv(x,y,lbda,sigmas)
  out_dict['gcv']['time'].append(time.time()-t1)
  out_dict['gcv']['sigma'].append(sigma_gcv)

  #LM
  t1=time.time()
  sigma_lm=log_marg(x,y,lbda,(1e-3,l))
  out_dict['lm']['time'].append(time.time()-t1)
  out_dict['lm']['sigma'].append(sigma_lm)

 # #JCV
 # t1=time.time()
 # sigmas1=np.logspace(np.log10(sigma_j/3),np.log10(3*sigma_j),n_sigmas)
 # sigma_jcv=gcv(x,y,lbda,sigmas1)
 # out_dict['jcv']['time'].append(time.time()-t1)
 # out_dict['jcv']['sigma'].append(sigma_jcv)

 # #JLM
 # t1=time.time()
 # sigma_jlm=log_marg_seed(x,y,lbda,sigma_j)
 # out_dict['jlm']['time'].append(time.time()-t1)
 # out_dict['jlm']['sigma'].append(sigma_jlm)


  y1_j=krr(x_test,x,y,sigma_j,lbda)
  y1_gcv=krr(x_test,x,y,sigma_gcv,lbda)
  y1_lm=krr(x_test,x,y,sigma_lm,lbda)
  y1_sil=krr(x_test,x,y,sigma_sil,lbda)
 # y1_jcv=krr(x_test,x,y,sigma_jcv,lbda)
 # y1_jlm=krr(x_test,x,y,sigma_jlm,lbda)

  out_dict['j']['r2'].append(r2(y_test,y1_j))
  out_dict['gcv']['r2'].append(r2(y_test,y1_gcv))
  out_dict['lm']['r2'].append(r2(y_test,y1_lm))
  out_dict['sil']['r2'].append(r2(y_test,y1_sil))
 # out_dict['jcv']['r2'].append(r2(y_test,y1_jcv))
 # out_dict['jlm']['r2'].append(r2(y_test,y1_jlm))

  krr(x_all,x_all,y_all,1,lbda) #To reset times
  
fi=open('data/french_1d'+suf+'.pkl','wb')
pickle.dump(out_dict,fi)
fi.close()

