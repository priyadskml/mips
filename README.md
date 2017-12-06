### mips<br />
ALSH for solving mips sublinearly<br />


################<br />
DATASET<br />
################<br />
Link: http://files.grouplens.org/datasets/movielens/ml-latest-small.zip<br />

To execute the code on a small dataset :<br />
1. run ./dataset_small.sh<br />
2. in ipynb files comment dataset = "datasets"+os.path.sep+"ml-latest" and uncomment dataset = "datasets"+os.path.sep+"ml-latest-small"<br />

To execute the code on full dataset :<br />
1. run ./dataset.sh
2. in ipynb files uncomment dataset = "datasets"+os.path.sep+"ml-latest" and comment dataset = "datasets"+os.path.sep+"ml-latest-small"<br />



################<br />
PAPER<br />
################<br />


Link to the paper : http://papers.nips.cc/paper/5329-asymmetric-lsh-alsh-for-sublinear-time-maximum-inner-product-search-mips.pdf<br />



#################<br />
CODE<br />
#################<br />
ALSH.mips has stand alone code for complete implementation of LSH, SLSH and ALSH (helper file: utils.py)<br />


#####<br />
Files: <br />
#####<br />
1. plots_paramas2.ipynb has functions for determining best parameters for U, m, r and c <br />
        It also has plots that show minimisations of rho functions.<br />
        
2. text.py has code to assemble netflix data from the zip file(s)<br />

3. utils.py has code for helper functions of L2LSH, SLSH and ALSH<br />

4. ALSH_MIPS.ipynb has code that tests on L2LSH, SLSH and ALSH.<br />
        It also has precision recall curves<br />

