# mips
ALSH for solving mips sublinearly

Link to the paper : http://papers.nips.cc/paper/5329-asymmetric-lsh-alsh-for-sublinear-time-maximum-inner-product-search-mips.pdf


ALSH.mips is a stand alone code that has complete implementation of LSH and ALSH(helper file: utils.py)

To execute the code on a small dataset :
1. run ./dataset_small.sh
2. in ipynb files comment dataset = "datasets"+os.path.sep+"ml-latest" and uncomment dataset = "datasets"+os.path.sep+"ml-latest-small"


To execute the code on full dataset :
1. run ./dataset.sh
2. in ipynb files uncomment dataset = "datasets"+os.path.sep+"ml-latest" and comment dataset = "datasets"+os.path.sep+"ml-latest-small"


Files: 
1. plots_paramas2.ipynb has functions for determining best parameters for U, m, r and c 
        It also has plots that show minimisations of rho functions.

2. text.py has code to assemble netflix data from the zip file(s)

3. utils.py has code for helper functions of L2LSH and ALSH

4. ALSH_MIPS.ipynb has code that tests on L2LSH and ALSH.
It also has precision recall curves

