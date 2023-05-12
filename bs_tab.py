import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from scipy.stats import wilcoxon
quant=0.10
data=pd.read_csv('bs_out.csv',sep=',')

lines=''
for alg, name in zip(['j','gcv','lm','jlm','sil'],['Jacobian','GCV','MML','Seeded MML','Silverman']):
  if alg=='j':
    line=name+' & '
  else:
    line='\\hline\n\\multirow{2}{*}{'+name+'} & '
  for metric in ['r2','time','sigma']:
    vec = data[metric+'s_'+alg].to_numpy()
    if metric=='time': vec*=1000
    #line+=f'${trim_mean(vec, quant):.3g}$ (${np.quantile(vec,quant):.3g}$, ${np.quantile(vec,1-quant):.3g}$)'
    line+=f'${np.mean(vec):.3g}$ (${np.quantile(vec,quant):.3g}$, ${np.quantile(vec,1-quant):.3g}$)'
    if metric=='sigma':
      line+=' \\\\\n'
    else:
      line+=' & '
  lines+=line

print(lines)
print('')
lines1=''
for alg in ['gcv','lm','jlm','sil']:
  line='& '
  for metric in ['r2','time']:
    if metric == 'r2':
      alt='greater'
    elif metric == 'time':
      alt='less'
    try:
      p_wil=wilcoxon(data[metric+'s_j'], data[metric+'s_'+alg], alternative=alt)[1]
    except:
      p_wil=1
    line+='$p_{\\text{Wil}}='+f'{p_wil:.2g}$ & '
    if metric=='time':
      line+=' \\\\\n'
  lines1+=line
  
print(lines1)


