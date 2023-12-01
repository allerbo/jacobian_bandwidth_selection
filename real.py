import numpy as np
import pandas as pd
import sys
sys.path.insert(1,'.')
from help_fcts import opt_sigma, r2, krr_map, gcv_map, get_sil_std, log_marg_map
import time
from sklearn import datasets
import pickle

data='energy'
seed=2
for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])


if data=='wood':
  dm=pd.read_csv('wood-fibres.csv',sep=',').to_numpy()
elif data=='casp':
  dm=pd.read_csv('CASP.csv',sep=',').to_numpy()
elif data=='bs':
  dm=pd.read_csv('bs.csv',sep=',').to_numpy()
elif data=='house':
  house=datasets.fetch_california_housing()
  dm=np.hstack((house.target.reshape((-1,1)),house.data))
elif data=='energy':
  dm=pd.read_csv('energydata.csv',sep=',').to_numpy()

#To reset times
opt_sigma(10,1,10,1e-3)
sigma_sil=get_sil_std(10,1,1)
gcv_map(np.array([[0,0.5],[0.5,0]]),np.array([0,1]).reshape((-1,1)),1e-3,[1,2])
log_marg_map(np.array([[0,0.5],[0.5,0]]),np.array([0,1]).reshape((-1,1)),1e-3,(1e-3,1))

data_dict={}
for metric in ['r2','time']:
  data_dict[metric]={}

lbda=1e-3
FRAC=0.65

np.random.seed(seed)
np.random.shuffle(dm)
dm=dm[:10000,:]
X_all=dm[:,1:]
X_all=(X_all-np.mean(X_all, 0))/np.std(X_all,0)
n_all=X_all.shape[0]
X=X_all[:int(FRAC*n_all),:]
X_test=X_all[int(FRAC*n_all):,:]
y_all=dm[:,0].reshape((-1,1))
y=y_all[:int(FRAC*n_all),:]
y_test=y_all[int(FRAC*n_all):,:]


n,p=X.shape
X2=np.sum(X**2,1).reshape((-1,1))
XX=X.dot(X.T)
D=np.sqrt(np.maximum(X2-2*XX+X2.T,0))
l=np.max(D)

XtX=X_test.dot(X.T)
Xt2=np.sum(X_test**2,1).reshape((-1,1))
D1=np.sqrt(np.maximum(Xt2-2*XtX+X2.T,0))


#Jacobian
t1=time.time()
sigma_j=opt_sigma(n,p,l,lbda)[0]
data_dict['time']['j']=time.time()-t1

#Silverman
t1=time.time()
sigma_sil=get_sil_std(n,p,np.std(X))
data_dict['time']['sil']=time.time()-t1

#GCV
t1=time.time()
sigmas=np.logspace(-3,np.log10(l),10)
sigma_gcv=gcv_map(D,y,lbda,sigmas)
data_dict['time']['gcv']=time.time()-t1

#MML
t1=time.time()
sigma_mml=log_marg_map(D,y,lbda,(1e-3,l))
data_dict['time']['mml']=time.time()-t1


#R2
y1_j=krr_map(D1,D,y,sigma_j,lbda)
data_dict['r2']['j']=r2(y_test,y1_j)
y1_sil=krr_map(D1,D,y,sigma_sil,lbda)
data_dict['r2']['sil']=r2(y_test,y1_sil)
y1_gcv=krr_map(D1,D,y,sigma_gcv,lbda)
data_dict['r2']['gcv']=r2(y_test,y1_gcv)
y1_mml=krr_map(D1,D,y,sigma_mml,lbda)
data_dict['r2']['mml']=r2(y_test,y1_mml)


fi=open('real_data/'+data+'_'+str(seed)+'.pkl','wb')
pickle.dump(data_dict,fi)
fi.close()

