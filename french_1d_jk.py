import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.model_selection import  KFold
from help_fcts import opt_sigma, r2, krr, get_sil, std_jk, gcv, cv_10, log_marg, log_marg_seed
from matplotlib import pyplot as plt
from datetime import datetime as dt

def plot_mean_std(ax,x_ax,list_o,c, z_ord, lw=1.5):
  mat=np.hstack(list_o)
  ax.plot(x_ax,np.mean(mat,1)-std_jk(mat,1),c+'--',linewidth=lw, zorder=z_ord)
  ax.plot(x_ax,np.mean(mat,1)+std_jk(mat,1),c+'--',linewidth=lw, zorder=z_ord)
  return ax.plot(x_ax,np.mean(mat,1),c,linewidth=lw, zorder=z_ord)

def plot_mean_std_sub(ax,x_ax,list_o,c, z_ord, sub, lw=2.5):
  mat=np.hstack(list_o)
  ax.plot(x_ax[:sub],np.mean(mat,1)[:sub],c,linewidth=lw, zorder=z_ord)
  ax.plot(x_ax[:sub],np.mean(mat,1)[:sub]-std_jk(mat,1)[:sub],c+'--',linewidth=lw, zorder=z_ord)
  ax.plot(x_ax[:sub],np.mean(mat,1)[:sub]+std_jk(mat,1)[:sub],c+'--',linewidth=lw, zorder=z_ord)


FS_LAB=15
FS_LEG=13

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

sigmas_j=[]
sigmas_cv=[]
sigmas_gcv=[]
sigmas_sil=[]
y1s_j=[]
y1s_cv=[]
y1s_gcv=[]
y1s_sil=[]
r2s_j=[]
r2s_cv=[]
r2s_gcv=[]
r2s_sil=[]
sigmas_lm=[]
y1s_lm=[]
r2s_lm=[]
sigmas_jlm=[]
y1s_jlm=[]
r2s_jlm=[]
print(len(x_train))
for i_del in range(len(x_train)):
  print(i_del)
  x=np.delete(x_train,i_del).reshape((-1,1))
  y=np.delete(y_train,i_del).reshape((-1,1))
  n=len(x)
  l=np.max(x)-np.min(x)
  
  #Jacobian
  sigma_j=opt_sigma(n,p,l,lbda)[0]
  sigmas_j.append(sigma_j)
  
  #Silverman
  sigma_sil=get_sil(n,p,x)
  sigmas_sil.append(sigma_sil)
  
  #GCV
  sigmas=np.logspace(-3,np.log10(l),100)
  sigma_gcv=gcv(x,y,lbda,sigmas)
  sigmas_gcv.append(sigma_gcv)
  
  #CV
  sigmas=np.logspace(-3,np.log10(l),100)
  sigma_cv=cv_10(x,y,lbda,sigmas,i_del)
  sigmas_cv.append(sigma_cv)
  
  #LM
  sigma_lm=log_marg(x,y,lbda,(1e-3,l))
  sigmas_lm.append(sigma_lm)
  
  #JLM
  sigma_jlm=log_marg_seed(x,y,lbda,sigma_j,(1e-3,l))
  sigmas_jlm.append(sigma_jlm)
  
  
  y1_j=krr(x1,x,y,sigma_j,lbda)
  y1_gcv=krr(x1,x,y,sigma_gcv,lbda)
  y1_cv=krr(x1,x,y,sigma_cv,lbda)
  y1_sil=krr(x1,x,y,sigma_sil,lbda)
  
  y1s_j.append(y1_j)
  y1s_gcv.append(y1_gcv)
  y1s_cv.append(y1_cv)
  y1s_sil.append(y1_sil)
  
  r2s_j.append(r2(y_test,krr(x_test,x,y,sigma_j,lbda)))
  r2s_gcv.append(r2(y_test,krr(x_test,x,y,sigma_gcv,lbda)))
  r2s_cv.append(r2(y_test,krr(x_test,x,y,sigma_cv,lbda)))
  r2s_sil.append(r2(y_test,krr(x_test,x,y,sigma_sil,lbda)))
  
  y1_lm=krr(x1,x,y,sigma_lm,lbda)
  y1s_lm.append(y1_lm)
  r2s_lm.append(r2(y_test,krr(x_test,x,y,sigma_lm,lbda)))
  
  y1_jlm=krr(x1,x,y,sigma_jlm,lbda)
  y1s_jlm.append(y1_jlm)
  r2s_jlm.append(r2(y_test,krr(x_test,x,y,sigma_jlm,lbda)))

fig=plt.figure(figsize=(13,7.2))
ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=3, rowspan=1)
ax1 = plt.subplot2grid((2, 3), (1, 0), colspan=2, rowspan=1)
ax2 = plt.subplot2grid((2, 3), (1, 2))


ls=[]

ls.append(ax0.plot(x_train,y_train,'ok',markersize=4,zorder=5)[0])
ls.append(ax0.plot(x_test,y_test,'ok', markersize=4,fillstyle='none',zorder=5)[0])
plot_mean_std(ax0,x1,y1s_j,'C2', 4)
plot_mean_std(ax0,x1,y1s_cv,'C1', 3)
plot_mean_std(ax0,x1,y1s_sil,'C3', 2)


ax0.set_ylabel('$^{\\circ}C$',fontsize=FS_LAB)

ax0.set_xlabel('Day',fontsize=FS_LAB)
days=np.arange(0,32,1)
ax0.set_xticks(days*24)
ax0.set_xticklabels(days)
ax0.set_xlim([-.3*24,31.3*24])
ax0.set_ylim([-11,23])

sub_train_idxs=np.argwhere(x_train<98)[:,0]
sub_test_idxs=np.argwhere(x_test<98)[:,0]
ax1.plot(x_train[sub_train_idxs],y_train[sub_train_idxs],'ok',markersize=10,zorder=5)
ax1.plot(x_test[sub_test_idxs],y_test[sub_test_idxs],'ok',markersize=10, fillstyle='none',zorder=5)
plot_mean_std_sub(ax1,x1,y1s_j,'C2', 4,132)
plot_mean_std_sub(ax1,x1,y1s_cv,'C1', 3,132)
plot_mean_std_sub(ax1,x1,y1s_sil,'C3', 2,132)


ax1.set_ylabel('$^{\\circ}C$',fontsize=FS_LAB)

ax1.set_xlabel('Day',fontsize=FS_LAB)
days1=np.arange(0,5,1)
ax1.set_xticks(days1*24)
ax1.set_xticklabels(days1)
ax1.set_xlim([-.1*24,4.1*24])
ax1.set_ylim([-5,15])




ax2.axis('off')
for c in ['C2','C1','C3']:
  ls.append(Line2D([0],[0],color=c,lw=1.5))

ax2.legend(ls, ['Training Data', 'Reference Data','Jacobian.\nMean$(\\sigma)=$'+str(np.round(np.mean(sigmas_j),2))+' h,\nStd$(\\sigma)=$'+str(np.round(std_jk(sigmas_j),3))+' h','Cross-validation.\nMean$(\\sigma)=$'+str(np.round(np.mean(sigmas_cv),2))+' h,\nStd$(\\sigma)=$'+str(np.round(std_jk(sigmas_cv),3))+' h', 'Silverman.\nMean$(\\sigma)=$'+str(np.round(np.mean(sigmas_sil),2))+' h,\nStd$(\\sigma)=$'+str(np.round(std_jk(sigmas_sil),2))+' h'], loc='lower center', ncol=1, fontsize=FS_LEG)
fig.tight_layout()

fig.savefig('figures/french_1d_jk.pdf')


fig=plt.figure(figsize=(13,7.2))
ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=3, rowspan=1)
ax1 = plt.subplot2grid((2, 3), (1, 0), colspan=2, rowspan=1)
ax2 = plt.subplot2grid((2, 3), (1, 2))

ls=[]

ls.append(ax0.plot(x_train,y_train,'ok',markersize=4,zorder=5)[0])
ls.append(ax0.plot(x_test,y_test,'ok', markersize=4,fillstyle='none',zorder=5)[0])
plot_mean_std(ax0,x1,y1s_j,'C2', 3, lw=2)
plot_mean_std(ax0,x1,y1s_lm,'C4', 2)
plot_mean_std(ax0,x1,y1s_jlm,'C0', 4, lw=1)


ax0.set_ylabel('$^{\\circ}C$',fontsize=FS_LAB)

ax0.set_xlabel('Day',fontsize=FS_LAB)
days=np.arange(0,32,1)
ax0.set_xticks(days*24)
ax0.set_xticklabels(days)
ax0.set_xlim([-.3*24,31.3*24])
ax0.set_ylim([-11,23])

sub_train_idxs=np.argwhere(x_train<98)[:,0]
sub_test_idxs=np.argwhere(x_test<98)[:,0]

ax1.plot(x_train[sub_train_idxs],y_train[sub_train_idxs],'ok',markersize=10,zorder=5)
ax1.plot(x_test[sub_test_idxs],y_test[sub_test_idxs],'ok',markersize=10, fillstyle='none',zorder=5)
plot_mean_std_sub(ax1,x1,y1s_j,'C2', 3,132, lw=4)
plot_mean_std_sub(ax1,x1,y1s_lm,'C4', 2,132)
plot_mean_std_sub(ax1,x1,y1s_jlm,'C0', 4,132, lw=2)


ax1.set_ylabel('$^{\\circ}C$',fontsize=FS_LAB)

ax1.set_xlabel('Day',fontsize=FS_LAB)
days1=np.arange(0,5,1)
ax1.set_xticks(days1*24)
ax1.set_xticklabels(days1)
ax1.set_xlim([-.1*24,4.1*24])
ax1.set_ylim([-5,15])




ax2.axis('off')
for c in ['C2','C4','C0']:
  ls.append(Line2D([0],[0],color=c,lw=1.5))

ax2.legend(ls, ['Training Data', 'Reference Data','Jacobian.\nMean$(\\sigma)=$'+str(np.round(np.mean(sigmas_j),2))+' h,\nStd$(\\sigma)=$'+str(np.round(std_jk(sigmas_j),3))+' h', 'Log Marginal.\nMean$(\\sigma)=$'+str(np.round(np.mean(sigmas_lm),2))+' h,\nStd$(\\sigma)=$'+str(np.round(std_jk(sigmas_lm),2))+' h', 'Jacobian Seeded LM.\nMean$(\\sigma)=$'+str(np.round(np.mean(sigmas_jlm),2))+' h,\nStd$(\\sigma)=$'+str(np.round(std_jk(sigmas_jlm),3))+' h'], loc='lower center', ncol=1, fontsize=FS_LEG)
fig.tight_layout()

fig.savefig('figures/french_1d_jk_lm.pdf')



