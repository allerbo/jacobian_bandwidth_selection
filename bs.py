import numpy as np
import pandas as pd
import sys
sys.path.insert(1,'.')
from help_fcts import opt_sigma, r2, krr_map, gcv_map, get_sil_std, log_marg_map, log_marg_map_seed
import time
#from scipy.stats import wilcoxon

#To reset times
opt_sigma(10,1,10,1e-3)
sigma_sil=get_sil_std(10,1,1)
gcv_map(np.eye(2),np.array([0,1]).reshape((-1,1)),1e-3,[0,1])
log_marg_map(np.eye(2),np.array([0,1]).reshape((-1,1)),1e-3,(1e-3,1))
##

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0



data=pd.read_csv('bs_2000.csv',sep=',').to_numpy()
days=np.unique(data[:,1])
MONTHS=[[1,31],[32,60],[61,91],[92,121],[122,152],[153,182],[183,213],[214,244],[245,274],[275,305],[306,335],[336,366]]

r2s_j=[]
r2s_gcv=[]
r2s_sil=[]
r2s_lm=[]
r2s_jlm=[]
sigmas_j=[]
sigmas_gcv=[]
sigmas_sil=[]
sigmas_lm=[]
sigmas_jlm=[]
times_j=[]
times_gcv=[]
times_sil=[]
times_lm=[]
times_jlm=[]

lbda=1e-3

ns=[]
for month in MONTHS:
  print(month)
  np.random.seed(0)
  data1=data[(data[:,1]>=month[0])&(data[:,1]<=month[1])]
  np.random.shuffle(data1)
  X_all=data1[:,(1,8,9)]
  X_all=(X_all-np.mean(X_all, 0))/np.std(X_all,0)
  y_all=data1[:,7].reshape((-1,1))
  n=X_all.shape[0]
  ns.append(n)
  X=X_all[:int(0.85*n),:]
  X_test=X_all[int(0.85*n):,:]
  y=y_all[:int(0.85*n),:]
  y_test=y_all[int(0.85*n):,:]


#  in_ch=in_hull(X_test,X)
#  X_test=X_test[in_ch,:]
#  y_test=y_test[in_ch,:]

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
  sigmas_j.append(sigma_j)
  times_j.append(time.time()-t1)

  #Silverman
  t1=time.time()
  sigma_sil=get_sil_std(n,p,np.std(X))
  sigmas_sil.append(sigma_sil)
  times_sil.append(time.time()-t1)

  #GCV
  t1=time.time()
  #sigmas=np.logspace(-3,np.log10(l),100)
  sigmas=np.logspace(-3,np.log10(l),10)
  sigma_gcv=gcv_map(D,y,lbda,sigmas)
  sigmas_gcv.append(sigma_gcv)
  times_gcv.append(time.time()-t1)

  #LM
  t1=time.time()
  sigma_lm=log_marg_map(D,y,lbda,(1e-3,l))
  times_lm.append(time.time()-t1)
  sigmas_lm.append(sigma_lm)

  #JLM
  t1=time.time()
  sigma_j=opt_sigma(n,p,l,lbda)[0]
  sigma_jlm=log_marg_map_seed(D,y,lbda,sigma_j)
  times_jlm.append(time.time()-t1)
  sigmas_jlm.append(sigma_jlm)


  #R2
  y1_j=krr_map(D1,D,y,sigma_j,lbda)
  r2s_j.append(r2(y_test,y1_j))
  y1_sil=krr_map(D1,D,y,sigma_sil,lbda)
  r2s_sil.append(r2(y_test,y1_sil))
  y1_gcv=krr_map(D1,D,y,sigma_gcv,lbda)
  r2s_gcv.append(r2(y_test,y1_gcv))
  y1_lm=krr_map(D1,D,y,sigma_lm,lbda)
  r2s_lm.append(r2(y_test,y1_lm))
  y1_jlm=krr_map(D1,D,y,sigma_jlm,lbda)
  r2s_jlm.append(r2(y_test,y1_jlm))
#  print(wilcoxon(r2s_j, r2s_sil, alternative='greater')[1], wilcoxon(r2s_j, r2s_gcv, alternative='greater')[1], wilcoxon(r2s_j, r2s_lm, alternative='greater')[1])

  out_dict={}
  for alg in ['j','sil','gcv','lm','jlm']:
    out_dict['r2s_'+alg]=   np.round(eval('r2s_'+alg),5)
  for alg in ['j','sil','gcv','lm','jlm']:
    out_dict['sigmas_'+alg]=np.round(eval('sigmas_'+alg),5)
  for alg in ['j','sil','gcv','lm','jlm']:
    out_dict['times_'+alg]= np.round(eval('times_'+alg),5)
  pd.DataFrame(out_dict).to_csv('bs_out.csv')

print(np.mean(ns))
