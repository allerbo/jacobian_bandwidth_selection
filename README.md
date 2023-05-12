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
python bw_sim_demo.py
```

## Figure 4
```
python french_2d_jk.py    
```

## Figure 5 and Figure 8
```
python french_1d_jk.py
```


## Figures 6
```
bash run_n.sh            #Calls start_synth_n.sh, start_french_1d_n.sh 
                         #and start_french_2d_n.sh multiple times for 
                         #different values of n. Each call of start_*_n.sh 
                         #starts an instance of *_n.py on a cluster.
python plot_sweep_n.py   #Plot output from above.
```

## Figures 7
```
bash run_lbda.sh           #Calls start_run_synth_lbda.sh, start_french_1d_lbda.sh 
                           #and start_french_2d_lbda.sh multiple times for 
                           #different values of n. Each call of start_*_lbda.sh 
                           #starts an instance of *_lbda.py on a cluster.
python plot_sweep_lbda.py  #Plot output from above.
```

## Table 1
```
python bs.py         #Run experiments
python bs_tab.py     #Create table
```

