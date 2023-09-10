# Info: This repo is forked  

This repo forked github repo of Official Github for SPMM and some files are newly added and adjusted specifically for tackling metabolism stability.  

# SPMM: Structure-Property Multi-Modal learning for molecules

Bidirectional Generation of Structure and Properties Through a Single Molecular Foundation Model.
https://arxiv.org/abs/2211.10590

***<ins>The model checkpoint and data are too heavy to be included in this repo, and they can be found [here](https://drive.google.com/drive/folders/1ARrSg9kXdXAL5VGgDBwizpSgcJwauPua?usp=sharing).<ins>***

![method1](https://github.com/jinhojsk515/SPMM/assets/59189526/1ff52950-aa12-481f-94ea-4d1e97ac7bf3)

Molecule structure will be given in SMILES, and we used 53 simple chemical properties to build a property vector(PV) of a molecule.

## Requirements
Run `pip install -r requirements.txt` to install the required packages.  

## Custom made regression task 

    ```
    python d_regression.py --checkpoint './Pretrain/checkpoint_SPMM_20m.ckpt' --name 'esol'
    python d_classification.py --checkpoint './Pretrain/checkpoint_SPMM_20m.ckpt' --name 'bbbp'
    python d_classification_multilabel.py --checkpoint './Pretrain/checkpoint_SPMM_20m.ckpt' --name 'clintox'
    ```

6. Forward/retro-reaction prediction tasks

    `d_rxn_prediction.py` performs both forward/reverse reaction prediction task on USPTO-480k and USPTO-50k dataset.

    e.g. forward reaction prediction, no beam search
    ```
    python d_rxn_prediction.py --checkpoint './Pretrain/checkpoint_SPMM_20m.ckpt' --mode 'forward' --n_beam 1 
    ```
    e.g. retro reaction prediction, beam search with k=3
    ```
    python d_rxn_prediction.py --checkpoint './Pretrain/checkpoint_SPMM_20m.ckpt' --mode 'retro' --n_beam 3 
    ```
