# MATIC
Official implementation of our MATIC model

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
