def std_jk(x, axis=None):
  import numpy as np
  if axis is None:
    n=len(x)
    return np.sqrt(n-1)*np.std(x)
  else:
    n=x.shape[axis]
    return np.sqrt(n-1)*np.std(x,axis)

def get_sil(n,p,X):
  import numpy as np
  return (4/(n*(p+2)))**(1/(p+4))*np.std(X)

def get_sil_std(n,p,std):
  import numpy as np
  return (4/(n*(p+2)))**(1/(p+4))*std

def calc_D_map(X,Xp):
  import numpy as np
  from pymap3d import vincenty
  D = np.zeros((X.shape[0],Xp.shape[0]))
  for i in range(X.shape[0]):
    for j in range(Xp.shape[0]):
      D[i,j]=vincenty.vdist(X[i,0], X[i,1], Xp[j,0], Xp[j,1])[0]
  return D

def krr_map(D1,D,y,sigma,lbda):
  import numpy as np
  K1=np.exp(-0.5*(D1/sigma)**2)
  K=np.exp(-0.5*(D/sigma)**2)
  #return K1.dot(np.linalg.inv(K+lbda*np.eye(K.shape[0]))).dot(y-np.mean(y))+np.mean(y)
  return K1@np.linalg.solve(K+lbda*np.eye(K.shape[0]),y-np.mean(y))+np.mean(y)

def krr(X1,X,y,sigma,lbda):
  import numpy as np
  return kern(X1,X,sigma)@np.linalg.solve(kern(X,X,sigma)+lbda*np.eye(X.shape[0]),y-np.mean(y))+np.mean(y)

def r2(y,y_hat):
  import numpy as np
  return 1-np.mean((y-y_hat)**2)/np.mean((y-np.mean(y))**2)

def j2a(sigma,n,p,l,lbda):
  import numpy as np
  d=2*l/(((n-1)**(1/p)-1)*np.pi)
  return 1/(n*sigma*np.exp(-(sigma/d)**2)+sigma*lbda), 1/sigma, 1/(n*np.exp(-(sigma/d)**2)+lbda)

def j2a_m(sigma,X,lbda):
  import numpy as np
  n=X.shape[0]
  X2=np.sum(X**2,1).reshape((-1,1))
  XX=X.dot(X.T)
  D=np.sqrt(X2-2*XX+X2.T)
  med_dist=np.median(np.min(D+10000*np.eye(n),0))
  d=2/np.pi*med_dist
  return 1/(n*sigma*np.exp(-(sigma/d)**2)+sigma*lbda), 1/sigma, 1/(n*np.exp(-(sigma/d)**2)+lbda)


def j2_emp(x,y,x1,lbda,sigmas):
  import numpy as np
  j2s_emp=[]
  for sigma in sigmas:
    y1=krr(x1,x,y,sigma,lbda)
    j2s_emp.append(np.max(np.abs(np.diff(y1,axis=0)/np.diff(x1,axis=0))))
  return np.array(j2s_emp)

def j2_emp_map(D,y,D1,lat_map,lon_map,lbda,sigmas):
  import numpy as np
  from pymap3d import vincenty
  n_width=lat_map.shape[0]
  Dx_0=np.zeros((n_width-1,n_width))
  for i in range(n_width-1):
    for j in range(n_width):
      Dx_0[i,j]=vincenty.vdist(lat_map[i,j], lon_map[i,j], lat_map[i+1,j], lon_map[i+1,j])[0]
  Dx_1=np.zeros((n_width,n_width-1))
  for i in range(n_width):
    for j in range(n_width-1):
      Dx_1[i,j]=vincenty.vdist(lat_map[i,j], lon_map[i,j], lat_map[i,j+1], lon_map[i,j+1])[0]
  j2s_emp=[]
  for sigma in sigmas:
    y1=krr_map(D1,D,y,sigma,lbda).reshape((n_width,n_width))
    dy1_0=np.diff(y1,axis=0)
    dy1_1=np.diff(y1,axis=1)
    j2s_emp.append(max(np.max(np.abs(dy1_0/Dx_0)), np.max(np.abs(dy1_1/Dx_1))))
  return np.array(j2s_emp)


def kern(X,Y,sigma):
  import numpy as np
  if X.shape[1]==1 and Y.shape[1]==1:
    return np.exp(-0.5*((X-Y.T)/sigma)**2)
  X2=np.sum(X**2,1).reshape((-1,1))
  XY=X.dot(Y.T)
  Y2=np.sum(Y**2,1).reshape((-1,1))
  D2=X2-2*XY+Y2.T
  return np.exp(-0.5*D2/sigma**2)

#def gcv_1d(x,y,lbda,sigmas):
#  import numpy as np
#  n=x.shape[0]
#  cvs=[]
#  for sigma in sigmas:
#    K= kern_1d(x,x,sigma)
#    yh=K@np.linalg.solve(K+lbda*np.eye(n),y-np.mean(y))+np.mean(y)
#    cvs.append(np.mean(((y-yh)/(1-np.trace(K+np.eye(n))/n))**2))
#  return sigmas[np.argmin(cvs)]

def gcv(X,y_in,lbda,sigmas):
  import numpy as np
  y=y_in-np.mean(y_in)
  n=X.shape[0]
  cvs=[]
  for sigma in sigmas:
    K_l=kern(X,X,sigma)+lbda*np.eye(n)
    K_li=np.linalg.inv(K_l)
    cvs.append(np.mean((K_li@y/np.diag(K_li).reshape((-1,1)))**2))
  return sigmas[np.argmin(cvs)]

def gcv_map(D,y_in,lbda,sigmas):
  import numpy as np
  y=y_in-np.mean(y_in)
  n=D.shape[0]
  cvs=[]
  for sigma in sigmas:
    K_l=np.exp(-0.5*(D/sigma)**2)+lbda*np.eye(n)
    K_li=np.linalg.inv(K_l)
    cvs.append(np.mean((K_li@y/np.diag(K_li).reshape((-1,1)))**2))
  return sigmas[np.argmin(cvs)]

def cv_n(X,y,lbda,sigmas):
  import numpy as np
  n=X.shape[0]
  mses_o=[]
  for sigma in sigmas:
    mses=[]
    for i in range(n):
      Xt=np.delete(X,i,axis=0)
      yt=np.delete(y,i,axis=0)
      Xv=X[i,:].reshape((-1,1))
      yv=y[i,:].reshape((-1,1))
      y1=krr(Xv,Xt,yt,sigma,lbda)
      mses.append(np.mean((yv-y1)**2))
    mses_o.append(mses)
  mses_mean=np.mean(np.array(mses_o),1)
  return sigmas[np.argmin(mses_mean)]

def cv_n_map(D,y,lbda,sigmas):
  import numpy as np
  n=D.shape[0]
  mses_o=[]
  for sigma in sigmas:
    mses=[]
    for i in range(n):
      yt=np.delete(y,i,axis=0)
      yv=y[i,:].reshape((1,-1))
      Dt=np.delete(np.delete(D,i,axis=0),i,axis=1)
      D1t=np.delete(D[i,:],i).reshape((1,-1))
      y1=krr_map(D1t,Dt,yt,sigma,lbda)
      mses.append(np.mean((yv-y1)**2))
    mses_o.append(mses)
  mses_mean=np.mean(np.array(mses_o),1)
  return sigmas[np.argmin(mses_mean)]

def cv_n_sk_1d(X,y,lbda,sigmas, seed, n_fold=10):
  import numpy as np
  from sklearn.model_selection import KFold
  n=X.shape[0]
  kf=KFold(n_splits=min(n,n_fold),shuffle=True, random_state=seed)
  mses_o=[]
  for sigma in sigmas:
    mses=[]
    for ti,vi in kf.split(X,y):
      Xt,Xv,yt,yv= X[ti,:], X[vi,:], y[ti], y[vi,:]
      y1=krr(Xv,Xt,yt,sigma,lbda)
      mses.append(np.mean((yv-y1)**2))
    mses_o.append(mses)
  mses_mean=np.mean(np.array(mses_o),1)
  return sigmas[np.argmin(mses_mean)]



def opt_sigma(n,p,l,lbda):
  import numpy as np
  from scipy.special import lambertw
  d=2*l/(((n-1)**(1/p)-1)*np.pi)
  w_arg=-lbda*np.exp(.5)/(2*n)
  if w_arg<-np.exp(-1):
    return d*np.sqrt(3/2), d*np.sqrt(3/2)
  w_0=np.real(lambertw(w_arg,k=0))
  w_1=np.real(lambertw(w_arg,k=-1))
  return d/np.sqrt(2)*np.sqrt(1-2*w_0), d/np.sqrt(2)*np.sqrt(1-2*w_1)

def opt_sigma_m(D,lbda):
  import numpy as np
  from scipy.special import lambertw
  n=D.shape[0]
  med_dist=np.median(np.min(D+10000*np.eye(n),0))
  d=2/np.pi*med_dist
  w_arg=-lbda*np.exp(.5)/(2*n)
  if w_arg<-np.exp(-1):
    return d*np.sqrt(3/2), d*np.sqrt(3/2)
  w_0=np.real(lambertw(w_arg,k=0))
  w_1=np.real(lambertw(w_arg,k=-1))
  return d/np.sqrt(2)*np.sqrt(1-2*w_0), d/np.sqrt(2)*np.sqrt(1-2*w_1)


def opt_sigma_m_map(D,lbda):
  import numpy as np
  from scipy.special import lambertw
  n=D.shape[0]
  med_dist=np.median(np.min(D+10000000*np.eye(n),0))
  d=2/np.pi*med_dist
  w_arg=-lbda*np.exp(.5)/(2*n)
  if w_arg<-np.exp(-1):
    return d*np.sqrt(3/2), d*np.sqrt(3/2)
  w_0=np.real(lambertw(w_arg,k=0))
  w_1=np.real(lambertw(w_arg,k=-1))
  return d/np.sqrt(2)*np.sqrt(1-2*w_0), d/np.sqrt(2)*np.sqrt(1-2*w_1)

def log_marg(X,y_in,lbda, sigma_bounds):
  import numpy as np
  from scipy.optimize import minimize_scalar
  y=y_in-np.mean(y_in)
  n=X.shape[0]
  def log_marg_fn(sigma):
    Kl=kern(X,X,sigma)+lbda*np.eye(n)
    return (y.T@np.linalg.solve(Kl,y) + np.log(np.linalg.det(Kl)))[0][0]
  
  res = minimize_scalar(log_marg_fn, bounds=sigma_bounds, method='bounded')
  return res.x

def log_marg_seed(X,y_in,lbda,sigma_seed, sigma_bounds):
  import numpy as np
  from scipy.optimize import minimize
  y=y_in-np.mean(y_in)
  n=X.shape[0]
  def log_marg_fn(sigmav):
    Kl=kern(X,X,sigmav[0])+lbda*np.eye(n)
    return (y.T@np.linalg.solve(Kl,y) + np.log(np.linalg.det(Kl)))[0][0]
  
  res = minimize(log_marg_fn, [sigma_seed], bounds=[sigma_bounds])
  return res.x[0]

def log_marg_map(D,y_in,lbda, sigma_bounds):
  import numpy as np
  from scipy.optimize import minimize_scalar
  y=y_in-np.mean(y_in)
  n=D.shape[0]
  def log_marg_fn(sigma):
    Kl=np.exp(-0.5*(D/sigma)**2)+lbda*np.eye(n)
    return (y.T@np.linalg.solve(Kl,y) + np.log(np.linalg.det(Kl)))[0][0]
  
  res = minimize_scalar(log_marg_fn, bounds=sigma_bounds, method='bounded')
  return res.x

def log_marg_map_seed(D,y_in,lbda, sigma_seed, sigma_bounds):
  import numpy as np
  from scipy.optimize import minimize
  y=y_in-np.mean(y_in)
  n=D.shape[0]
  def log_marg_fn(sigmav):
    Kl=np.exp(-0.5*(D/sigmav[0])**2)+lbda*np.eye(n)
    return (y.T@np.linalg.solve(Kl,y) + np.log(np.linalg.det(Kl)))[0][0]
  
  res = minimize(log_marg_fn, [sigma_seed], bounds=[sigma_bounds])
  return res.x[0]


def cv_10(X,y_in,lbda,sigmas,seed):
  import numpy as np
  y=y_in-np.mean(y_in)
  n=X.shape[0]
  np.random.seed(seed)
  per=np.random.permutation(n)
  folds=np.array_split(per,10)
  mses_o=[]
  for sigma in sigmas:
    mses=[]
    for v_fold in range(len(folds)):
      t_folds=np.concatenate([folds[t_fold] for t_fold in range(len(folds)) if v_fold != t_fold])
      v_folds=folds[v_fold]
      Xt=X[t_folds,:]
      yt=y[t_folds,:]
      Xv=X[v_folds,:]
      yv=y[v_folds,:]
      y1=krr(Xv,Xt,yt,sigma,lbda)
      mses.append(np.mean((yv-y1)**2))
    mses_o.append(mses)
  mses_mean=np.mean(np.array(mses_o),1)
  return sigmas[np.argmin(mses_mean)]


def cv_10_map(D,y_in,lbda,sigmas,seed):
  import numpy as np
  y=y_in-np.mean(y_in)
  n=D.shape[0]
  np.random.seed(seed)
  per=np.random.permutation(n)
  folds=np.array_split(per,10)
  mses_o=[]
  for sigma in sigmas:
    mses=[]
    for v_fold in range(len(folds)):
      t_folds=np.concatenate([folds[t_fold] for t_fold in range(len(folds)) if v_fold != t_fold])
      v_folds=folds[v_fold]
      Dt=D[np.ix_(t_folds,t_folds)]
      yt=y[t_folds,:]
      D1vt=D[np.ix_(v_folds,t_folds)]
      yv=y[v_folds,:]
      y1=krr_map(D1vt,Dt,yt,sigma,lbda)
      mses.append(np.mean((yv-y1)**2))
    mses_o.append(mses)
  mses_mean=np.mean(np.array(mses_o),1)
  return sigmas[np.argmin(mses_mean)]
