import numpy as np
import sys
sys.path.insert(1,'.')
from help_fcts import opt_sigma, r2, krr, get_sil
from sklearn.model_selection import KFold

def f(X):
  y=np.sin(2*np.pi*X[:,0])
  if X.shape[1]>1:
    for ip in range(1,p):
      y*= np.sin(2*np.pi*X[:,ip])
  return y.reshape((-1,1))



N=1001
p=1
lbda=1e-3
n=40

noise=0.1

r2s_0_o=[]
r2s_cv_o=[]
r2s_sil_o=[]
sigmas_0_o=[]
sigmas_cv_o=[]
sigmas_sil_o=[]

r2s_cv1_o=[]
sigmas_cv1_o=[]

l=10
N_SEEDS=1000

log10lbda=int(sys.argv[1])
lbda=10**(0.1*log10lbda)
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
  X=np.random.uniform(-l/2,l/2,(n,p))
  y=f(X)+np.random.normal(0,noise,(n,1))
  
  X_test=np.linspace(-l/2,l/2,N).reshape((-1,1))
  y_test=f(X_test)
  
  l1=np.max(X)-np.min(X)
   
  #Jacobian
  sigma0=opt_sigma(n,p,l1,lbda)[0]
  sigmas_0.append(sigma0)
  
  #CV
  kf=KFold(n_splits=min(n,10),shuffle=True, random_state=seed)
  sigmas=np.logspace(-2,np.log10(l1),100)
  mses_o=[]
  for sigma in sigmas:
    mses=[]
    for ti,vi in kf.split(X,y):
      Xt,Xv,yt,yv= X[ti,:], X[vi,:], y[ti], y[vi,:]
      y1=krr(Xv,Xt,yt,sigma,lbda)
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
    for ti,vi in kf.split(X,y):
      Xt,Xv,yt,yv= X[ti,:], X[vi,:], y[ti], y[vi,:]
      y1=krr(Xv,Xt,yt,sigma,lbda)
      mses.append(np.mean((yv-y1)**2))
    mses_o.append(mses)
  
  mses_mean=np.mean(np.array(mses_o),1)
  sigma_cv1=sigmas[np.argmin(mses_mean)]
  sigmas_cv1.append(sigma_cv1)
  
  ##Silverman
  sigma_sil=get_sil(n,p,X)
  sigmas_sil.append(sigma_sil)
  
  y1_0=krr(X_test,X,y,sigma0,lbda)
  y1_cv=krr(X_test,X,y,sigma_cv,lbda)
  y1_sil=krr(X_test,X,y,sigma_sil,lbda)
  r2s_0.append(r2(f(X_test),y1_0))
  r2s_cv.append(r2(f(X_test),y1_cv))
  r2s_sil.append(r2(f(X_test),y1_sil))
  
  y1_cv1=krr(X_test,X,y,sigma_cv1,lbda)
  r2s_cv1.append(r2(f(X_test),y1_cv1))
  
np.save('data/synth_r2s_0_lbda_'+str(log10lbda),r2s_0)
np.save('data/synth_r2s_cv_lbda_'+str(log10lbda),r2s_cv)
np.save('data/synth_r2s_sil_lbda_'+str(log10lbda),r2s_sil)
np.save('data/synth_sigmas_0_lbda_'+str(log10lbda),sigmas_0)
np.save('data/synth_sigmas_cv_lbda_'+str(log10lbda),sigmas_cv)
np.save('data/synth_sigmas_sil_lbda_'+str(log10lbda),sigmas_sil)

np.save('data/synth_r2s_cv1_lbda_'+str(log10lbda),r2s_cv1)
np.save('data/synth_sigmas_cv1_lbda_'+str(log10lbda),sigmas_cv1)
