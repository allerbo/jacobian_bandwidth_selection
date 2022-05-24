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
  return K1.dot(np.linalg.inv(K+lbda*np.eye(K.shape[0]))).dot(y-np.mean(y))+np.mean(y)

  mat=np.vstack(list_o)
  if km:
    mat*=0.001
  if with_quants:
    ax.plot(x_ax,np.quantile(mat,quant,1),c+'--')
    ax.plot(x_ax,np.quantile(mat,1-quant,1),c+'--')
  return ax.plot(x_ax,np.mean(mat,1),c)

def krr(X1,X,y,sigma,lbda):
  import numpy as np
  return kern(X1,X,sigma).dot(np.linalg.inv(kern(X,X,sigma)+lbda*np.eye(X.shape[0]))).dot(y-np.mean(y))+np.mean(y)

def r2(y,y_hat):
  import numpy as np
  return 1-np.mean((y-y_hat)**2)/np.mean((y-np.mean(y))**2)

def j2a(sigma,n,p,l,lbda):
  import numpy as np
  d=2*l/(((n-1)**(1/p)-1)*np.pi)
  return 1/(n*sigma*np.exp(-(sigma/d)**2)+sigma*lbda), 1/sigma, 1/(n*np.exp(-(sigma/d)**2)+lbda)

def kern(X,Y,sigma):
  import numpy as np
  X2=np.sum(X**2,1).reshape((-1,1))
  XY=X.dot(Y.T)
  Y2=np.sum(Y**2,1).reshape((-1,1))
  D2=X2-2*XY+Y2.T
  return np.exp(-0.5*D2/sigma**2)

def kern_1d(x,y,sigma):
  import numpy as np
  return np.exp(-0.5*((x-y.T)/sigma)**2)

def opt_sigma(n,p,l,lbda,ver=0):
  import numpy as np
  from scipy.special import lambertw
  if ver==1:
    d=2*l/((n-1)**(1/p)*np.pi)
  else:
    d=2*l/(((n-1)**(1/p)-1)*np.pi)
  w_arg=-lbda*np.exp(.5)/(2*n)
  if w_arg<-np.exp(-1):
    return d*np.sqrt(3/2), d*np.sqrt(3/2)
  w_0=np.real(lambertw(w_arg,k=0))
  w_1=np.real(lambertw(w_arg,k=-1))
  return d/np.sqrt(2)*np.sqrt(1-2*w_0), d/np.sqrt(2)*np.sqrt(1-2*w_1)

