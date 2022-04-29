import os
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from metric import accuracy, precision, recall, mcc
from AttentiveFP import save_smiles_dicts, get_smiles_array
from model import GAT, mmoe, matic
from dataloader import load_data, MolCollator
from utils import param_count
from collections import OrderedDict
from utils import BinaryFocalLoss

def run_training(args, logger):
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.
    """
    torch.cuda.set_device(args.gpu)

    train_data, val_data, test_data = load_data(args, logger)

    feature_dicts = save_smiles_dicts(train_data.smiles()+val_data.smiles()+test_data.smiles())

    save_dir = os.path.join(args.save_dir, 'model')
    os.makedirs(save_dir, exist_ok=True)

    if args.num_tasks==1:
        model = GAT(3, 1, 39, 10, 150, 0.1)
        logger.debug(f'build single task model')
        if args.checkpoints:
            logger.debug(f'load pretrained single task model from ' + str(args.checkpoints))
            pretrained_state_dict = torch.load(args.checkpoints)
            new_state_dict = OrderedDict()
            for (k, v) in pretrained_state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
    else:
        model = matic()
        logger.debug(f'build multi task matic model')
        if args.checkpoints:
            logger.debug(f'load pretrained single task model from ' + str(args.checkpoints))
            model.load_state_dict(torch.load(args.checkpoints))

    logger.debug(model)
    logger.debug(f'Number of parameters = {param_count(model):,}')

    if args.loss == "bce":
        logger.debug(f'create bce loss')
        loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        logger.debug(f'create focal loss')
        loss_func = BinaryFocalLoss(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model = model.cuda()

    # Run training
    best_epoch = 0
    min_val_loss = float('inf')
    for epoch in range(args.epochs):
        logger.debug(f'-----------------------------------')
        train_scores, train_loss = train(
            epoch=epoch,
            model=model,
            data=train_data,
            loss_func=loss_func,
            optimizer=optimizer,
            args=args,
            logger=logger,
            feature_dicts = feature_dicts
        )

        val_scores, val_loss = evaluate(
            model=model,
            data=val_data,
            loss_func=loss_func,
            logger=logger,
            args=args,
            feature_dicts = feature_dicts
        )

        logger.debug(f'Epoch: {epoch:03d}, loss_train: {train_loss:.3f}, loss_val: {val_loss:.3f}')

        for task_name, train_score in zip(args.task_names, train_scores):
            logger.debug(f'Train {task_name} auc : {train_score[0]:.3f}, accuracy : {train_score[1]:.3f}, precision : {train_score[2]:.3f}, recall : {train_score[3]:.3f}, mcc : {train_score[4]:.3f}')

        for task_name, val_score in zip(args.task_names, val_scores):
            logger.debug(f'Validation {task_name} auc : {val_score[0]:.3f}, accuracy : {val_score[1]:.3f}, precision : {val_score[2]:.3f}, recall : {val_score[3]:.3f}, mcc : {val_score[4]:.3f}')

        # Save model checkpoint if improved validation score
        if val_loss < min_val_loss:
            min_val_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

        if epoch - best_epoch > args.early_stop_epoch:
            break

    logger.info(f'Model best val loss = {min_val_loss:.3f} on epoch {best_epoch}')

    if args.num_tasks==1:
        test_model = GAT(3, 1, 39, 10, 150, 0.1)
        logger.debug(f'test single task model')
    else:
        test_model = matic()
        logger.debug(f'test multi matic task model')
        # test_model = mmoe()
        # logger.debug(f'test multi mmoe task model')

    test_model.load_state_dict(torch.load(os.path.join(save_dir, 'model.pt')))
    test_model.cuda()
    logger.debug('load best model to test')

    test_scores, test_loss = evaluate(
        model=test_model,
        data=test_data,
        loss_func=loss_func,
        logger=logger,
        args=args,
        feature_dicts = feature_dicts
    )

    for task_name, test_score in zip(args.task_names, test_scores):
        logger.debug(f'Test {task_name} auc : {test_score[0]:.3f}, accuracy : {test_score[1]:.3f}, precision : {test_score[2]:.3f}, recall : {test_score[3]:.3f}, mcc : {test_score[4]:.3f}')


def train(epoch, model, data, loss_func, optimizer, args, logger, feature_dicts):

    model.train()

    loss_sum, total_batch = 0, 0

    mol_collator = MolCollator(args=args)
    mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, collate_fn=mol_collator)

    total_preds = []
    total_targets = []
    for _, item in enumerate(mol_loader):

        smiles_batch, mask, targets = item
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, _ = get_smiles_array(smiles_batch, feature_dicts)
        x_atom, x_bonds, x_mask = torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.Tensor(x_mask)
        x_atom_index,x_bond_index = torch.LongTensor(x_atom_index), torch.LongTensor(x_bond_index)
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = x_atom.cuda(), x_bonds.cuda(),x_atom_index.cuda(),x_bond_index.cuda(), x_mask.cuda()
        mask, targets = mask.cuda(), targets.cuda()

        # Run model
        model.zero_grad()
        if args.num_tasks ==2:
            class_weights = torch.ones(targets.shape)
            class_weights[:,0] = args.class_weights
            class_weights[:,1] = 1 - args.class_weights
        else:
            class_weights = torch.ones(targets.shape)
        class_weights = class_weights.cuda()

        batch_preds, attention, _, _ , aux_out = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
        aux_preds = torch.ones(targets.shape)
        aux_preds[1, :] = 0
        loss = loss_func(batch_preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()
        loss_aux = loss_func(aux_out.cuda(), aux_preds.cuda()).mean()
        loss = loss - loss_aux * 0.3

        loss_sum += loss.item()
        total_batch += 1

        loss.backward()
        optimizer.step()

        batch_preds = F.sigmoid(batch_preds).data.cpu().numpy().tolist()
        batch_targets = targets.data.cpu().numpy().tolist()
        total_preds.extend(batch_preds)
        total_targets.extend(batch_targets)

    loss_avg = loss_sum / total_batch

    train_preds = [[] for _ in range(args.num_tasks)]
    train_targets = [[] for _ in range(args.num_tasks)]
    for i in range(args.num_tasks):
        for j in range(len(total_preds)):
            if total_targets[j][i] is not None:  # Skip those without targets
                train_preds[i].append(total_preds[j][i])
                train_targets[i].append(total_targets[j][i])

    # Compute metric
    results = []
    for i in range(args.num_tasks):
        results_tmp = []
        results_tmp.append(roc_auc_score(train_targets[i], train_preds[i]))
        results_tmp.append(accuracy(train_targets[i], train_preds[i]))
        results_tmp.append(precision(train_targets[i], train_preds[i]))
        results_tmp.append(recall(train_targets[i], train_preds[i]))
        results_tmp.append(mcc(train_targets[i], train_preds[i]))
        results.append(results_tmp)

    return results, loss_avg


def evaluate(model, data, loss_func, args, logger, feature_dicts):

    model.eval()

    preds = []
    loss_sum, total_batch = 0, 0

    mol_collator = MolCollator(args=args)
    mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            collate_fn=mol_collator)

    for _, item in enumerate(mol_loader):
        # _, batch, features_batch, mask, targets = item
        smiles_batch, mask, targets = item
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, _ = get_smiles_array(smiles_batch, feature_dicts)
        x_atom, x_bonds, x_mask = torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.Tensor(x_mask)
        x_atom_index,x_bond_index = torch.LongTensor(x_atom_index), torch.LongTensor(x_bond_index)
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = x_atom.cuda(), x_bonds.cuda(),x_atom_index.cuda(),x_bond_index.cuda(), x_mask.cuda()
        mask, targets = mask.cuda(), targets.cuda()

        if args.num_tasks ==2:
            class_weights = torch.ones(targets.shape)
            class_weights[:,0] = args.class_weights
            class_weights[:,1] = 1 - args.class_weights
        else:
            class_weights = torch.ones(targets.shape)
        class_weights = class_weights.cuda()

        with torch.no_grad():
            batch_preds, attention, _, _, _ = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
            total_batch += 1

            loss = loss_func(batch_preds, targets) * class_weights * mask
            loss = loss.sum() / mask.sum()
            loss_sum += loss.item()

            batch_preds = F.sigmoid(batch_preds).data.cpu().numpy().tolist()
            preds.extend(batch_preds)

    loss_avg = loss_sum / total_batch

    targets = data.targets()

    valid_preds = [[] for _ in range(args.num_tasks)]
    valid_targets = [[] for _ in range(args.num_tasks)]
    for i in range(args.num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric
    results = []
    for i in range(args.num_tasks):
        results_tmp = []
        results_tmp.append(roc_auc_score(valid_targets[i], valid_preds[i]))
        results_tmp.append(accuracy(valid_targets[i], valid_preds[i]))
        results_tmp.append(precision(valid_targets[i], valid_preds[i]))
        results_tmp.append(recall(valid_targets[i], valid_preds[i]))
        results_tmp.append(mcc(valid_targets[i], valid_preds[i]))
        results.append(results_tmp)

    return results, loss_avg