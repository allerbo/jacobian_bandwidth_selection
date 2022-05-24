import numpy as np
import pandas as pd
from sklearn.model_selection import  KFold
from help_fcts import opt_sigma, r2, krr, get_sil, std_jk
from matplotlib import pyplot as plt
from datetime import datetime as dt

def plot_mean_std(ax,x_ax,list_o,c, z_ord):
  mat=np.hstack(list_o)
  ax.plot(x_ax,np.mean(mat,1)-std_jk(mat,1),c+'--',linewidth=1.5, zorder=z_ord)
  ax.plot(x_ax,np.mean(mat,1)+std_jk(mat,1),c+'--',linewidth=1.5, zorder=z_ord)
  return ax.plot(x_ax,np.mean(mat,1),c,linewidth=1.5, zorder=z_ord)

def plot_mean_std_sub(ax,x_ax,list_o,c, z_ord, sub):
  mat=np.hstack(list_o)
  ax.plot(x_ax[:sub],np.mean(mat,1)[:sub],c,linewidth=2.5, zorder=z_ord)
  ax.plot(x_ax[:sub],np.mean(mat,1)[:sub]-std_jk(mat,1)[:sub],c+'--',linewidth=2.5, zorder=z_ord)
  ax.plot(x_ax[:sub],np.mean(mat,1)[:sub]+std_jk(mat,1)[:sub],c+'--',linewidth=2.5, zorder=z_ord)


FS_LAB=13
FS_LEG=12

temps_data = pd.read_csv('french_1d.csv', delimiter=";")

y_all = temps_data[['t']].values-273.15
x_temp = temps_data[['date']].values
x_temp1=list(map(lambda d: dt.strptime(str(d)[1:11],'%Y%m%d%H'),x_temp))
x_all=np.array(list(map(lambda d: (d-x_temp1[0]).total_seconds()/3600,x_temp1))).reshape((-1,1))

x1=np.linspace(np.min(x_all),np.max(x_all),1001).reshape((-1,1))



n=len(x_all)//2
np.random.seed(1)
per=np.random.permutation(len(x_all))
x_train, x_test=x_all[per[:n]], x_all[per[n:]]
y_train, y_test=y_all[per[:n]], y_all[per[n:]]

lbda=1e-3
p=1

sigmas_0=[]
sigmas_cv=[]
sigmas_sil=[]
y1s_0=[]
y1s_cv=[]
y1s_sil=[]
r2s_0=[]
r2s_cv=[]
r2s_sil=[]
for i_del in range(len(x_train)):
  x=np.delete(x_train,i_del).reshape((-1,1))
  y=np.delete(y_train,i_del).reshape((-1,1))
  n=len(x)
  l=np.max(x)-np.min(x)
  
  #Jacobian
  sigma0=opt_sigma(n,p,l,lbda)[0]
  sigmas_0.append(sigma0)
  
  #Silverman
  sigma_sil=get_sil(n,p,x)
  sigmas_sil.append(sigma_sil)
  
  #CV
  kf=KFold(n_splits=min(n,10),shuffle=True, random_state=i_del)
  sigmas=np.logspace(-2,np.log10(l),10)
  mses_o=[]
  for sigma in sigmas:
    mses=[]
    for ti,vi in kf.split(x,y):
      xt,xv,yt,yv= x[ti], x[vi], y[ti], y[vi]
      y1=krr(xv,xt,yt,sigma,lbda)
      mses.append(np.mean((yv-y1)**2))
    mses_o.append(mses)
  
  mses_mean=np.mean(np.array(mses_o),1)
  sigma_cv=sigmas[np.argmin(mses_mean)]
  sigmas_cv.append(sigma_cv)
  
  y1_0=krr(x1,x,y,sigma0,lbda)
  y1_cv=krr(x1,x,y,sigma_cv,lbda)
  y1_sil=krr(x1,x,y,sigma_sil,lbda)
  
  y1s_0.append(y1_0)
  y1s_cv.append(y1_cv)
  y1s_sil.append(y1_sil)
  
  r2s_0.append(r2(y_test,krr(x_test,x,y,sigma0,lbda)))
  r2s_cv.append(r2(y_test,krr(x_test,x,y,sigma_cv,lbda)))
  r2s_sil.append(r2(y_test,krr(x_test,x,y,sigma_sil,lbda)))



fig=plt.figure(figsize=(13,9))
ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=3, rowspan=1)
ax1 = plt.subplot2grid((2, 3), (1, 0), colspan=2, rowspan=1)
ax2 = plt.subplot2grid((2, 3), (1, 2))

ls=[]

ls.append(ax0.plot(x_train,y_train,'ok',markersize=4,zorder=5)[0])
ls.append(ax0.plot(x_test,y_test,'ok', markersize=4,fillstyle='none',zorder=5)[0])
ls.append(plot_mean_std(ax0,x1,y1s_0,'C2', 4)[0])
ls.append(plot_mean_std(ax0,x1,y1s_cv,'C1', 3)[0])
ls.append(plot_mean_std(ax0,x1,y1s_sil,'C3', 2)[0])


ax0.set_ylabel('$^{\\circ}C$',fontsize=FS_LAB)

ax0.set_xlabel('Day',fontsize=FS_LAB)
days=np.arange(0,32,1)
ax0.set_xticks(days*24)
ax0.set_xticklabels(days)
ax0.set_xlim([-.3*24,31.3*24])
ax0.set_ylim([-20,30])

sub_train_idxs=np.argwhere(x_train<98)[:,0]
sub_test_idxs=np.argwhere(x_test<98)[:,0]
ax1.plot(x_train[sub_train_idxs],y_train[sub_train_idxs],'ok',markersize=10,zorder=5)
ax1.plot(x_test[sub_test_idxs],y_test[sub_test_idxs],'ok',markersize=10, fillstyle='none',zorder=5)
plot_mean_std_sub(ax1,x1,y1s_0,'C2', 4,132)
plot_mean_std_sub(ax1,x1,y1s_cv,'C1', 3,132)
plot_mean_std_sub(ax1,x1,y1s_sil,'C3', 2,132)


ax1.set_ylabel('$^{\\circ}C$',fontsize=FS_LAB)

ax1.set_xlabel('Day',fontsize=FS_LAB)
days1=np.arange(0,5,1)
ax1.set_xticks(days1*24)
ax1.set_xticklabels(days1)
ax1.set_xlim([-.1*24,4.1*24])
ax1.set_ylim([-4.3,12.8])




ax2.axis('off')
ax2.legend(ls, ['Training Data', 'Reference Data','Jacobian.\nMean$(\\sigma)=$'+str(np.round(np.mean(sigmas_0),2))+' h, Std$(\\sigma)=$'+str(np.round(std_jk(sigmas_0),3))+' h','Cross-validation\nMean$(\\sigma)=$'+str(np.round(np.mean(sigmas_cv),2))+' h, Std$(\\sigma)=$'+str(np.round(std_jk(sigmas_cv),3))+' h', 'Silverman\nMean$(\\sigma)=$'+str(np.round(np.mean(sigmas_sil),2))+' h, Std$(\\sigma)=$'+str(np.round(std_jk(sigmas_sil),2))+' h'], loc='lower center', ncol=1, fontsize=FS_LEG)
plt.tight_layout()

plt.savefig('figures/french_1d_jk.pdf')

