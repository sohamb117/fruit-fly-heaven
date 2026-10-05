# drosophila_wing_cpg

Minimal code to produce figures in the manuscript "Gap junctions desynchronize a neural circuit to stabilize insect flight".


## Install packages
To create an environment with the packages necessary to run this code use
```bash
conda env create -f drosophila_mini.yml
```

To activate the conda environment use
```bash
conda activate drosophila_mini
```
In the utils we use a reduced version of `brianutils` (no additional installation required). You can find the full `brianutils` project [here](https://itbgit.biologie.hu-berlin.de/compneurophys/brianutils).

## Generate figures
To run simulations and generate the figures in the manuscript, run the following in a bash.
If you want to reduce run time, you can run the "sim" scripts in parallel.

```bash
python py/F3B_sim_exampletrace.py
python py/F3C_sim_varyggap.py	  
python py/F3D_sim_weakstrongGJ.py
python py/F3E_simana_couplingfunctions.py
python py/F3H_sim_hetSNLSNIC.py
python py/F3J_sim_hethom.py

python py/F3C_ana_varyggap.py
python py/F3D_ana_syncsshakB.py	 
python py/F3H_ana_syncshab.py
python py/F3J_ana_hethom.py

python py/F3B_plot.py		 
python py/F3C_plot.py		 
python py/F3D_plot_syncsshakB.py
python py/F3H_plot_syncshab.py		
python py/F3J_plot.py

python py/F3E_plot_couplingfunctions.py
```
