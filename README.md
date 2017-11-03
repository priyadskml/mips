# mips
ALSH for solving mips sublinearly

Link to the paper : http://papers.nips.cc/paper/5329-asymmetric-lsh-alsh-for-sublinear-time-maximum-inner-product-search-mips.pdf




To execute the code on a small dataset :
1. run ./dataset_small.sh
2. in ipynb files comment dataset = "datasets"+os.path.sep+"ml-latest" and uncomment dataset = "datasets"+os.path.sep+"ml-latest-small"


To execute the code on full dataset :
1. run ./dataset.sh
2. in ipynb files uncomment dataset = "datasets"+os.path.sep+"ml-latest" and comment dataset = "datasets"+os.path.sep+"ml-latest-small"
