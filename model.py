import torch
import torch.nn as nn
import torch.nn.functional as F

from AttentiveFP import Fingerprint

class GAT(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim, fingerprint_dim, p_dropout, num_task = 1):
        super(GAT, self).__init__()

        self.GAT = Fingerprint(radius, T, input_feature_dim, input_bond_dim, \
                               fingerprint_dim, p_dropout)

        self.predict = nn.Sequential(nn.Dropout(p_dropout),
                                        nn.Linear(fingerprint_dim, num_task))
        
    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):
        smile_feature, mol_attention, fea_relu, fea = self.GAT(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)

        prediction = self.predict(smile_feature)

        return prediction, mol_attention, fea_relu, fea

class expert1(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim, fingerprint_dim, p_dropout, num_task = 1):
        super(expert1, self).__init__()

        self.GAT = Fingerprint(radius, T, input_feature_dim, input_bond_dim, \
                               fingerprint_dim, p_dropout)
        
    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):

        smile_feature, mol_attention, fea_relu, fea = self.GAT(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)

        return smile_feature, mol_attention, fea_relu, fea

class gate(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim, fingerprint_dim, p_dropout, num_task = 1):
        super(gate, self).__init__()

        self.GAT = Fingerprint(radius, T, input_feature_dim, input_bond_dim, \
                               fingerprint_dim, p_dropout)

        self.dnn = nn.Sequential(nn.Dropout(p_dropout),
                                 nn.Linear(fingerprint_dim, 2))
        
    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):

        smile_feature, mol_attention, _, _ = self.GAT(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)
        output = self.dnn(smile_feature)

        return output

class expert2(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim, fingerprint_dim, p_dropout, num_task = 1):
        super(expert2, self).__init__()

        self.GAT = Fingerprint(radius, T, input_feature_dim, input_bond_dim, \
                               fingerprint_dim, p_dropout)

        self.dnn = nn.Sequential(nn.Dropout(p_dropout),
                                 nn.Linear(fingerprint_dim, 3))
        
    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):

        smile_feature, mol_attention, _, _ = self.GAT(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)
        output = self.dnn(smile_feature)

        return output

class Tower(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class matic(nn.Module):
    def __init__(self):
        super(matic, self).__init__()
        self.experts_shared = expert1(3, 1, 39, 10, 150, 0.1)
        self.experts_task1 = expert1(3, 1, 39, 10, 150, 0.1)
        self.experts_task2 = expert1(3, 1, 39, 10, 150, 0.1)
        self.gate1 = gate(3, 1, 39, 10, 150, 0.1)
        self.gate2 = gate(3, 1, 39, 10, 150, 0.1)
        self.tower1 = Tower(150, 1, 32)
        self.tower2 = Tower(150, 1, 32)


    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):
        experts_shared_o, experts_shared_att, share_f1, share_f2 = self.experts_shared(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)
        experts_task1_o, experts_task1_att, task1_f1, task1_f2 = self.experts_task1(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)
        experts_task2_o, experts_task2_att, task2_f1, task2_f2 = self.experts_task2(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)

        # gate1
        selected1 = self.gate1(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)
        selected1 = F.softmax(selected1,dim = 1)
        gate_expert_output1 = torch.stack([experts_task1_o, experts_shared_o])
        gate1_out = torch.einsum('abc, ba -> bc', gate_expert_output1, selected1)
        final_output1 = self.tower1(gate1_out)

        # gate2
        selected2 = self.gate2(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)
        selected2 = F.softmax(selected2,dim = 1)
        gate_expert_output2 = torch.stack([experts_task2_o, experts_shared_o])
        gate2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected2)
        final_output2 = self.tower2(gate2_out)

        out = torch.cat([final_output1, final_output2], dim = 1)
        att = [experts_shared_att, experts_task1_att, experts_task2_att, selected1, selected2]
        fea_relu = [share_f1, task1_f1, task2_f1]
        fea = [share_f2, task1_f2, task2_f2]
        return out, att, fea_relu, fea

class mmoe(nn.Module):
    def __init__(self):
        super(mmoe, self).__init__()
        self.experts = nn.ModuleList([expert1(3, 1, 39, 10, 150, 0.4) for i in range(3)])
        self.gate1 = expert2(3, 1, 39, 10, 150, 0.4)
        self.gate2 = expert2(3, 1, 39, 10, 150, 0.4)
        self.tower1 = Tower(150, 1, 32)
        self.tower2 = Tower(150, 1, 32)


    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):
        experts_result = []
        for e in self.experts:
            expert_o, expert_att, _, _ = e(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)
            experts_result.append(expert_o)
        experts_result = torch.stack(experts_result)

        # gate1
        selected1 = self.gate1(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)
        selected1 = F.softmax(selected1,dim = 1)
        gate1_out = torch.einsum('abc, ba -> bc', experts_result, selected1)
        final_output1 = self.tower1(gate1_out)

        # gate2
        selected2 = self.gate2(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)
        selected2 = F.softmax(selected2,dim = 1)
        gate2_out = torch.einsum('abc, ba -> bc', experts_result, selected2)
        final_output2 = self.tower2(gate2_out)

        out = torch.cat([final_output1, final_output2], dim = 1)

        return out, "", "", ""
