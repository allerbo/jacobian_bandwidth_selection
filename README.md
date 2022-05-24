This is the code used in the article **Bandwidth Selection for Gaussian Kernel Ridge Regression via Jacobian Control**.

## Download French Temperature Data:
```
python get_french.py         #Create files french_1d.csv and french_2d.csv.
                             #(No need since files are included.)
```

## Figure 1:
```
python bandwidth_demo.py 
```

## Figures 2:
```
python appr_jac_demo.py
```

## Figure 3
```
python french_1d_jk.py
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
bash run_synth_n.sh          #Call start_run_synth_n.sh multiple times for 
                             #different values of n. Each call of start_synth_n.sh 
                             #starts an instance of synth_n.py on a cluster.
bash run_synth_lbda.sh       #Equivalent to run_synth_n.sh, for sweeping lambda.
bash run_french_1d_n.sh      #Equivalent to run_synth_n.sh.
bash run_french_1d_lbda.sh   #Equivalent to run_synth_lbda.sh.
bash run_french_2d_n.sh      #Equivalent to run_synth_n.sh.
bash run_french_2d_lbda.sh   #Equivalent to run_synth_lbda.sh.
python plot_sweep_all.py     #Plot output from above.
                             #Set JAC_SEED=True to include Jacobian seeded 
                             #cross-validation.
```
