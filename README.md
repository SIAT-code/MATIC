# MATIC
Implementation of paper:

Hu, F., Wang, D., Huang, H., Hu, Y., & Yin, P. (2022). Bridging the Gap between Target-Based and Cell-Based Drug Discovery with a Graph Generative Multitask Model. Journal of Chemical Information and Modeling. https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c01180

## Requirement
```
python 3.7
pytorch 1.4
rdkit 2020.03.3
numpy 1.18.1
pandas 1.0.1
scikit-learn  0.23.2
```

## Usage 
* ```model``` contains the code implementation of MATIC model 
* ```data/``` contains some example data
* Using the main.py script to train the model. For example:
```
python main.py  --train_path data/examples/multi_train.csv \
                --val_path data/examples/multi_val.csv \
                --test_path data/examples/multi_test.csv \
                --save_dir model/ \
                --batch_size 64 \
                --epochs 100 \
                --init_lr 0.00015 \
                --early_stop_epoch 20\
                --gpu 0
```


## Use agreement
The SOFTWARE will be used for teaching or not-for-profit research purposes only. Permission is required for any commerical use of the Software.
