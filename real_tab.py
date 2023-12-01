import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import os
import pickle
quant=0.10

def to_latex(x):
  x_str=f'{x:.2g}'
  if 'e' in x_str:
    base, exponent=x_str.split('e')
    if exponent[0]=='+': exponent=exponent[-2:]
    exponent=exponent.replace('0','')
    if exponent=='2':
      return str(round(float(base)*10)*10)
    return base+'\\cdot 10^{'+exponent+'}'
  return x_str
data_names=['\\makecell{Aspen\\\\Fibres}', '\\makecell{Appliances\\\\Energy Use}','\\makecell{California\\\\Housing}','\\makecell{Protein\\\\Structure}','\\makecell{U.K.\\ Black\\\\Smoke}']
data_files = ['wood','energy','house','casp','bs']
alg_names=['Jacobian','GCV','MML','Silverman']
algs = ['j','gcv','mml','sil']
metrics = ['r2','time']
seeds=range(100)

tab_dict={}
for data_file in data_files:
  tab_dict[data_file]={}
  for alg in algs:
    tab_dict[data_file][alg]={}
    for metric in metrics:
      tab_dict[data_file][alg][metric]=[]


for data_file in data_files:
  for seed in seeds:
    if os.path.exists('real_data/'+data_file+'_'+str(seed)+'.pkl'):
      fi=open('real_data/'+data_file+'_'+str(seed)+'.pkl','rb')
      data_dict_seed=pickle.load(fi)
      fi.close()
      for alg in algs:
        for metric in metrics:
          tab_dict[data_file][alg][metric].append(data_dict_seed[metric][alg])




mean_lines=[]
q19_lines=[]
p_lines=[]
for data_file, data_name in zip(data_files,data_names):
  #data=pd.read_csv(data_file+'_out.csv',sep=',')
  for alg, alg_name in zip(algs,alg_names):
    if alg=='j':
      mean_line='\\hline\n \\multirow{7}{*}{'+data_name+'} & '+alg_name+' '
    else:
      mean_line='\\cline{2-4}\n & \\multirow{2}{*}{'+alg_name+'} '
    q19_line=' & '
    p_line=' & '
    for metric in metrics:
      vec = tab_dict[data_file][alg][metric]
      mean_line+=f' & $'+to_latex(np.mean(vec))+'$ ($'+to_latex(np.quantile(vec,quant))+'$, $'+to_latex(np.quantile(vec,1-quant))+'$)'
      if alg=='j':
        p_line=''
      else:
        if metric == 'r2':
          alt='greater'
        elif metric == 'time':
          alt='less'
        try:
          p_wil=wilcoxon(tab_dict[data_file]['j'][metric], vec, alternative=alt)[1]
        except:
          p_wil=1
        if p_wil<=0.01:
          p_line+='& $p_{\\text{Wil}}=\\bm{'+to_latex(p_wil)+'}$'
        else:
          p_line+='& $p_{\\text{Wil}}='+to_latex(p_wil)+'$'
    mean_lines.append(mean_line)
    q19_lines.append(q19_line)
    p_lines.append(p_line)

for mean_line, q19_line, p_line in zip(mean_lines, q19_lines, p_lines):
  print(mean_line+ '\\\\')
  #print(q19_line+ '\\\\')
  if p_line!='':
    print(p_line+ '\\\\')

