#!/usr/bin/env python
from __future__ import print_function, division
import sys 
sys.path.append("..")
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit.Chem.QED as QED
import scripts.sascorer as sascorer


import torch
import torch.nn.functional as F
from AttentiveFP import save_smiles_dicts, get_smiles_array
from predictor import GAT, PLE

torch.cuda.set_device(0)

rdBase.DisableLog('rdApp.error')

# class reward_model():
#     # 单任务模型gat
#     def __init__(self, model_path):
#         self.model = GAT(3, 1, 39, 10, 150, 0.1)
#         self.model.load_state_dict(torch.load(model_path))
#         self.model.cuda()

#     def __call__(self, smiles):
#         scores = []
#         with torch.no_grad():
#             for s in smiles:
#                 mol = Chem.MolFromSmiles(s)
#                 if mol is None:
#                     scores.append(0)
#                 else:
#                     feature_dicts = save_smiles_dicts([s])
#                     x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, _ = get_smiles_array([s], feature_dicts)
#                     x_atom, x_bonds, x_mask = torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.Tensor(x_mask)
#                     x_atom_index,x_bond_index = torch.LongTensor(x_atom_index), torch.LongTensor(x_bond_index)
#                     x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = x_atom.cuda(), x_bonds.cuda(),x_atom_index.cuda(),x_bond_index.cuda(), x_mask.cuda()
#                     batch_preds = self.model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
#                     score = F.sigmoid(batch_preds).data.cpu().numpy()
#                     scores.append(score[0][0])

#         return np.array(scores, dtype=np.float32)

class reward_model():
    #多任务模型ple
    def __init__(self, model_path):
        self.model = PLE()
        self.model.load_state_dict(torch.load(model_path))
        self.model.cuda()

    def __call__(self, smiles):
        scores_3cl = []
        scores_cell = []
        valid = 0
        with torch.no_grad():
            for s in smiles:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    scores_3cl.append(0)
                    scores_cell.append(0)
                else:
                    valid = valid+1
                    try:
                        feature_dicts = save_smiles_dicts([s])
                        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, _ = get_smiles_array([s], feature_dicts)
                        x_atom, x_bonds, x_mask = torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.Tensor(x_mask)
                        x_atom_index,x_bond_index = torch.LongTensor(x_atom_index), torch.LongTensor(x_bond_index)
                        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = x_atom.cuda(), x_bonds.cuda(),x_atom_index.cuda(),x_bond_index.cuda(), x_mask.cuda()
                        batch_preds = self.model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
                        score = F.sigmoid(batch_preds).data.cpu().numpy()
                        scores_3cl.append(score[0][0])
                        scores_cell.append(score[0][1])
                    except:
                        scores_3cl.append(0)
                        scores_cell.append(0)
        return np.array(scores_3cl, dtype=np.float32), np.array(scores_cell, dtype=np.float32)
        # return np.array(scores_3cl, dtype=np.float32), np.array(scores_cell, dtype=np.float32), valid / len(smiles)

class qed_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(0)
            else:
                scores.append(QED.qed(mol))
        return np.float32(scores)


class sa_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(10)
            else:
                scores.append(sascorer.calculateScore(mol))
        return np.float32(scores)


def get_scoring_function(prop_name):
    """Function that initializes and returns a scoring function by name"""
    if  prop_name =='3cl_cell':
        return reward_model("/home/wdq/multi_task/ple/model/1102split_fold0_ple_3clweight05/model/model.pt")
    elif prop_name == 'qed':
        return qed_func()
    elif prop_name == 'sa':
        return sa_func()
    else:
        return None

if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--prop', required=True)

    args = parser.parse_args()
    funcs = [get_scoring_function(prop) for prop in args.prop.split(',')]

    data = [line.strip() for line in sys.stdin]
    def scoring_function(x):
        p_3cl, p_cell  = funcs[0](x)
        if len(funcs)>1:
            p_qed = funcs[1](x)
            p_sa = funcs[2](x)
            # print('有效率:',  valid)
            return [p_3cl,p_cell, p_qed, p_sa]
        else:
            return [p_3cl, p_cell]
    # props = [func(data) for func in funcs]
    props = scoring_function(data)

    col_list = [data]+props
    for tup in zip(*col_list):
        print(*tup)



    # all_x, all_y = zip(*data)
    # def scoring_function(x):
    #     p_3cl, p_cell = funcs[0](x)
    #     if len(funcs)>1:
    #         p_qed = funcs[1](x)
    #         p_sa = funcs[2](x)
    #         # print(p_3cl.mean(), p_cell.mean(), p_qed.mean(), p_sa.mean(), valid)
    #         return [p_3cl,p_cell, p_qed, p_sa]
    #     else:
    #         return [p_3cl, p_cell]
    # # props = [func(data) for func in funcs]
    # props = scoring_function(all_y)

    # col_list = [all_x, all_y]+props
    # for tup in zip(*col_list):
    #     print(*tup)
