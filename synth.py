import numpy as np
import sys
import time
sys.path.insert(1,'.')
from help_fcts import opt_sigma, r2, krr, get_sil, gcv, cv_10, opt_sigma_m, log_marg, log_marg_seed

def f(X):
  y=np.sin(2*np.pi*X[:,0])
  if X.shape[1]>1:
    for ip in range(1,p):
      y*= np.sin(2*np.pi*X[:,ip])
  return y.reshape((-1,1))



N=1001
p=1

noise=0.2

N_SEEDS=1000

suf='_c'
if len(sys.argv)>=4:
  for arg in range(3,len(sys.argv)):
    exec(sys.argv[arg])

tp=sys.argv[1]

if tp=='n':
  lbda=1e-3
  n=int(sys.argv[2])
  suf1='_n_'+str(n)
elif tp=='lbda':
  n=50
  log10lbda=int(sys.argv[2])
  lbda=10**(0.1*log10lbda)
  suf1='_lbda_'+str(log10lbda)
else:
  print('Wrong type')
  sys.exit()

r2s_j=[]
r2s_jm=[]
r2s_cv=[]
r2s_gcv=[]
r2s_jcv=[]
r2s_sil=[]
sigmas_j=[]
sigmas_jm=[]
sigmas_cv=[]
sigmas_gcv=[]
sigmas_jcv=[]
sigmas_sil=[]
times_j=[]
times_jm=[]
times_gcv=[]
times_jcv=[]
times_cv=[]
times_sil=[]
r2s_lm=[]
sigmas_lm=[]
times_lm=[]
r2s_jlm=[]
sigmas_jlm=[]
times_jlm=[]

for seed in range(N_SEEDS):
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
  times_j.append(time.time()-t1)
  sigmas_j.append(sigma_j)

  #Jacobian M
  X2=np.sum(X**2,1).reshape((-1,1))
  XX=X.dot(X.T)
  D=np.sqrt(X2-2*XX+X2.T)
  t1=time.time()
  sigma_jm=opt_sigma_m(D,lbda)[0]
  times_jm.append(time.time()-t1)
  sigmas_jm.append(sigma_jm)
  
  sigmas=np.logspace(-3,np.log10(l1),100)
  #CV
  t1=time.time()
  sigma_cv=cv_10(X,y,lbda,sigmas,seed)
  times_cv.append(time.time()-t1)
  sigmas_cv.append(sigma_cv)

  #GCV
  t1=time.time()
  sigma_gcv=gcv(X,y,lbda,sigmas)
  times_gcv.append(time.time()-t1)
  sigmas_gcv.append(sigma_gcv)

  #JCV
  sigmas1=np.logspace(np.log10(sigma_jm/3),np.log10(3*sigma_jm),100)
  t1=time.time()
  sigma_jcv=cv_10(X,y,lbda,sigmas1,seed)
  times_jcv.append(time.time()-t1)
  sigmas_jcv.append(sigma_jcv)

  ##Silverman
  t1=time.time()
  sigma_sil=get_sil(n,p,X)
  times_sil.append(time.time()-t1)
  sigmas_sil.append(sigma_sil)

  #LM
  t1=time.time()
  sigma_lm=log_marg(X,y,lbda,(1e-3,l1))
  times_lm.append(time.time()-t1)
  sigmas_lm.append(sigma_lm)

  #JLM
  t1=time.time()
  sigma_jlm=log_marg_seed(X,y,lbda,sigma_jm,(1e-3,l1))
  times_jlm.append(time.time()-t1)
  sigmas_jlm.append(sigma_jlm)
  
  y1_j=krr(X_test,X,y,sigma_j,lbda)
  y1_jm=krr(X_test,X,y,sigma_jm,lbda)
  y1_gcv=krr(X_test,X,y,sigma_gcv,lbda)
  y1_sil=krr(X_test,X,y,sigma_sil,lbda)
  r2s_j.append(r2(f(X_test),y1_j))
  r2s_jm.append(r2(f(X_test),y1_jm))
  r2s_gcv.append(r2(f(X_test),y1_gcv))
  r2s_sil.append(r2(f(X_test),y1_sil))

  y1_cv=krr(X_test,X,y,sigma_cv,lbda)
  r2s_cv.append(r2(f(X_test),y1_cv))
  y1_jcv=krr(X_test,X,y,sigma_jcv,lbda)
  r2s_jcv.append(r2(f(X_test),y1_jcv))
  y1_lm=krr(X_test,X,y,sigma_lm,lbda)
  r2s_lm.append(r2(f(X_test),y1_lm))
  y1_jlm=krr(X_test,X,y,sigma_jlm,lbda)
  r2s_jlm.append(r2(f(X_test),y1_jlm))
  
  krr(X_test, X_test, f(X_test),1,lbda) #To reset times
  
np.save('data_synth'+suf+'/synth'+suf+'_r2s_j'+suf1,r2s_j)
np.save('data_synth'+suf+'/synth'+suf+'_r2s_jm'+suf1,r2s_jm)
np.save('data_synth'+suf+'/synth'+suf+'_r2s_gcv'+suf1,r2s_gcv)
np.save('data_synth'+suf+'/synth'+suf+'_r2s_sil'+suf1,r2s_sil)

np.save('data_synth'+suf+'/synth'+suf+'_sigmas_j'+suf1,sigmas_j)
np.save('data_synth'+suf+'/synth'+suf+'_sigmas_jm'+suf1,sigmas_jm)
np.save('data_synth'+suf+'/synth'+suf+'_sigmas_gcv'+suf1,sigmas_gcv)
np.save('data_synth'+suf+'/synth'+suf+'_sigmas_sil'+suf1,sigmas_sil)

np.save('data_synth'+suf+'/synth'+suf+'_times_j'+suf1,times_j)
np.save('data_synth'+suf+'/synth'+suf+'_times_jm'+suf1,times_jm)
np.save('data_synth'+suf+'/synth'+suf+'_times_gcv'+suf1,times_gcv)
np.save('data_synth'+suf+'/synth'+suf+'_times_sil'+suf1,times_sil)

np.save('data_synth'+suf+'/synth'+suf+'_r2s_cv'+suf1,r2s_cv)
np.save('data_synth'+suf+'/synth'+suf+'_sigmas_cv'+suf1,sigmas_cv)
np.save('data_synth'+suf+'/synth'+suf+'_times_cv'+suf1,times_cv)

np.save('data_synth'+suf+'/synth'+suf+'_r2s_jcv'+suf1,r2s_jcv)
np.save('data_synth'+suf+'/synth'+suf+'_sigmas_jcv'+suf1,sigmas_jcv)
np.save('data_synth'+suf+'/synth'+suf+'_times_jcv'+suf1,times_jcv)

np.save('data_synth'+suf+'/synth'+suf+'_r2s_lm'+suf1,r2s_lm)
np.save('data_synth'+suf+'/synth'+suf+'_sigmas_lm'+suf1,sigmas_lm)
np.save('data_synth'+suf+'/synth'+suf+'_times_lm'+suf1,times_lm)
np.save('data_synth'+suf+'/synth'+suf+'_r2s_jlm'+suf1,r2s_jlm)
np.save('data_synth'+suf+'/synth'+suf+'_sigmas_jlm'+suf1,sigmas_jlm)
np.save('data_synth'+suf+'/synth'+suf+'_times_jlm'+suf1,times_jlm)

