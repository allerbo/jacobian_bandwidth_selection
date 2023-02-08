This is the code used in the article **Bandwidth Selection for Gaussian Kernel Ridge Regression via Jacobian Control**, availible at https://arxiv.org/abs/2205.11956.

## Download French Temperature Data:
```
python get_french.py         #Create files french_1d.csv and french_2d.csv.
                             #(No need since files are included.)
```

## Figure 1:
```
python bandwidth_demo.py 
```

## Figure 2:
```
python appr_jac_demo.py
```

## Figure 3:
```
python french_2d_demo.py
```

## Figure 4
```
bash run_french_2d_jk.sh     #Call start_french_2d_jk.sh multiple times. 
                             #Each call of start_french_2d_jk.sh starts
                             #an instance of french_2d_jk.py on a cluster.
python plot_french_2d_jk.py  #Plot output from french_2d_jk.py
```

## Figure 5
```
python french_1d_jk.py
```


## Figures 6 and 7
```
bash run_synth_n.sh          #Call start_run_synth_n.sh multiple times for 
                             #different values of n. Each call of start_synth_n.sh 
                             #starts an instance of synth.py on a cluster.
bash run_french_1d_n.sh      #Equivalent to run_synth_n.sh.
bash run_french_2d_n.sh      #Equivalent to run_synth_n.sh.
python plot_sweep_n.py       #Plot output from above into Figure 6
python plot_sweep_n_jcv.py   #Plot output from above into Figure 7
```

## Figures in Supplementary Materials
```
bash run_synth_n.sh suf=\"_u\"     #Uniform distribution
bash run_synth_n.sh suf=\"_n\"     #Normal distribution
bash run_synth_n.sh suf=\"_e\"     #Exponential distribution
bash run_synth_lbda.sh
bash run_synth_lbda.sh suf=\"_u\"
bash run_synth_lbda.sh suf=\"_n\"
bash run_synth_lbda.sh suf=\"_e\"
bash run_french_1d_lbda.sh 
bash run_french_2d_lbda.sh 
python plot_french_2d_jk_lm.py     #Figure 1
python french_1d_jk.py             #Figure 2
python plot_sweep_n_lm.py          #Figure 3
python plot_sweep_n_seeded.py      #Figure 4
python plot_sweep_n_syn.py         #Figure 5
python plot_sweep_lbda.py          #Figure 6
python plot_sweep_lbda_syn.py      #Figure 7
python plot_sweep_n_syn_jcv.py     #Figure 8
python plot_sweep_lbda_jcv.py      #Figure 9
python plot_sweep_lbda_syn_jcv.py  #Figure 10
python french_1d_demo.py           #Figure 11
python synth_demo.py               #Figure 12
```
