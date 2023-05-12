import numpy as np
import sys
import time
sys.path.insert(1,'.')
from help_fcts import opt_sigma, r2, krr, get_sil, gcv, cv_10, opt_sigma_m, log_marg, log_marg_seed
import pickle

def f(X):
  y=np.sin(2*np.pi*X[:,0])
  if X.shape[1]>1:
    for ip in range(1,p):
      y*= np.sin(2*np.pi*X[:,ip])
  return y.reshape((-1,1))



N=1001
p=1

noise=0.2

suf='_c'
if len(sys.argv)>=3:
  for arg in range(2,len(sys.argv)):
    exec(sys.argv[arg])

n_sigmas=100
seed=int(sys.argv[1])
n=50
suf1='_lbda_'+str(seed)

out_dict={}
for alg in ['j','jm','gcv','lm','sil']:#,'jcv','jlm']:
  out_dict[alg]={}
  for metric in ['r2','sigma','time']:
    out_dict[alg][metric]=[]

for lbda in np.logspace(-5,2,15):
  np.random.seed(seed)
  if suf=='_c':
    sig=3
    X=sig*np.random.standard_cauchy((n,p))
    X_test=sig*np.random.standard_cauchy((N,p))
  elif suf=='_u':
    l=20
    X=np.random.uniform(-l/2,l/2,(n,p))
    X_test=np.random.uniform(-l/2,l/2,(N,p))
  elif suf=='_n':
    sig=10
    X=np.random.normal(0,sig,(n,p))
    X_test=np.random.normal(0,sig,(N,p))
  elif suf=='_e':
    bet=10
    X=np.random.exponential(bet,(n,p))
    X_test=np.random.exponential(bet,(N,p))
 # elif suf=='_b':
 #   X=np.random.uniform(-50,50,(n,p))
 #   N_SEEDS=100

  y=f(X)+np.random.normal(0,noise,(n,p))
  
  
  l1=np.max(X)-np.min(X)
   
  #Jacobian
  t1=time.time()
  sigma_j=opt_sigma(n,p,l1,lbda)[0]
  out_dict['j']['time'].append(time.time()-t1)
  out_dict['j']['sigma'].append(sigma_j)

  #Jacobian M
  X2=np.sum(X**2,1).reshape((-1,1))
  XX=X.dot(X.T)
  D=np.sqrt(X2-2*XX+X2.T)
  t1=time.time()
  sigma_jm=opt_sigma_m(D,lbda)[0]
  out_dict['jm']['time'].append(time.time()-t1)
  out_dict['jm']['sigma'].append(sigma_jm)
  
  ##Silverman
  t1=time.time()
  sigma_sil=get_sil(n,p,X)
  out_dict['sil']['time'].append(time.time()-t1)
  out_dict['sil']['sigma'].append(sigma_sil)

  #GCV
  sigmas=np.logspace(-3,np.log10(l1),n_sigmas)
  t1=time.time()
  sigma_gcv=gcv(X,y,lbda,sigmas)
  out_dict['gcv']['time'].append(time.time()-t1)
  out_dict['gcv']['sigma'].append(sigma_gcv)

  #LM
  t1=time.time()
  sigma_lm=log_marg(X,y,lbda,(1e-3,l1))
  out_dict['lm']['time'].append(time.time()-t1)
  out_dict['lm']['sigma'].append(sigma_lm)

 # #JCV
 # sigmas1=np.logspace(np.log10(sigma_jm/3),np.log10(3*sigma_jm),n_sigmas)
 # t1=time.time()
 # sigma_jcv=gcv(X,y,lbda,sigmas1)
 # out_dict['jcv']['time'].append(time.time()-t1)
 # out_dict['jcv']['sigma'].append(sigma_jcv)

 # #JLM
 # t1=time.time()
 # sigma_jlm=log_marg_seed(X,y,lbda,sigma_jm)
 # out_dict['jlm']['time'].append(time.time()-t1)
 # out_dict['jlm']['sigma'].append(sigma_jlm)
  
  y1_j=krr(X_test,X,y,sigma_j,lbda)
  y1_jm=krr(X_test,X,y,sigma_jm,lbda)
  y1_gcv=krr(X_test,X,y,sigma_gcv,lbda)
  y1_lm=krr(X_test,X,y,sigma_lm,lbda)
  y1_sil=krr(X_test,X,y,sigma_sil,lbda)
 # y1_jcv=krr(X_test,X,y,sigma_jcv,lbda)
 # y1_jlm=krr(X_test,X,y,sigma_jlm,lbda)

  out_dict['j']['r2'].append(r2(f(X_test),y1_j))
  out_dict['jm']['r2'].append(r2(f(X_test),y1_jm))
  out_dict['gcv']['r2'].append(r2(f(X_test),y1_gcv))
  out_dict['lm']['r2'].append(r2(f(X_test),y1_lm))
  out_dict['sil']['r2'].append(r2(f(X_test),y1_sil))
 # out_dict['jcv']['r2'].append(r2(f(X_test),y1_jcv))
 # out_dict['jlm']['r2'].append(r2(f(X_test),y1_jlm))

  
  krr(X_test, X_test, f(X_test),1,lbda) #To reset times

fi=open('data/synth'+suf+suf1+'.pkl','wb')
pickle.dump(out_dict,fi)
fi.close()

