import numpy as np
import copy
import torch
import dgl
from utils import set_seed
from train_and_eval_utils.utils import print_debug_info, early_stop_counter, print_debug_info_inductive
from train_and_eval_utils.gnn import train,evaluate,eval_on_train_val_test_data_gnn,eval_on_val_test_inductive_gnn
"""
1. Train and eval
"""

def train_sage(model, dataloader, feats, labels, criterion, optimizer, lamb=1):
    """
    Train for GraphSAGE. Process the graph in mini-batches using `dataloader` instead the entire graph `g`.
    lamb: weight parameter lambda
    """
    device = feats.device
    model.train()
    total_loss = 0
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]

        # Compute loss and prediction
        logits = model(blocks, batch_feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out, batch_labels)
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)




def get_train_eval_datasets_sage(g,idx_train,batch_size,fan_out,num_workers):
    # Create dataloader for SAGE

    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    g.create_formats_()
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [eval(fanout) for fanout in fan_out.split(",")]
    )
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        idx_train,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighors
    sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader_eval = dgl.dataloading.NodeDataLoader(
        g,
        torch.arange(g.num_nodes()),
        sampler_eval,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    data = dataloader
    data_eval = dataloader_eval
    return data,data_eval


def run_transductive_sage(
    conf,
    model,
    g,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    idx_train, idx_val, idx_test = indices
    batch_size = conf["batch_size"]
    data,data_eval= get_train_eval_datasets_sage(g,idx_train,batch_size,conf["fan_out"],conf["num_workers"])
    best_epoch, best_score_val, count,state = 0, 0, 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, conf["max_epoch"] + 1):
        loss = train_sage(model, data, feats, labels, criterion, optimizer)
        if epoch % conf["eval_interval"] == 0:
            (
                loss_train, 
                score_train,
                loss_val,
                score_val,
                loss_test,
                score_test
            )=eval_on_train_val_test_data_gnn(model, data_eval, feats, labels, criterion, evaluator, idx_train,idx_val,idx_test)
            print_debug_info(epoch,loss, loss_train,loss_val,loss_test,score_train,score_val,score_test,logger, loss_and_score)
            best_epoch, best_score_val, count,state=early_stop_counter(epoch,model,score_val, best_epoch, best_score_val, count,state)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break
    model.load_state_dict(state)
    out, _, score_val = evaluate(
        model, data_eval, feats, labels, criterion, evaluator, idx_val
    )
    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test

def run_inductive_sage(    
    conf,
    model,
    obs_g,
    g,
    obs_feats, 
    obs_labels,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score
):
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
    batch_size = conf["batch_size"]
    obs_data,obs_data_eval= get_train_eval_datasets_sage(obs_g,obs_idx_train,batch_size,conf["fan_out"],conf["num_workers"])
    data,data_eval=get_train_eval_datasets_sage(g,obs_idx_train,batch_size,conf["fan_out"],conf["num_workers"])

    best_epoch, best_score_val, count,state = 0, 0, 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, conf["max_epoch"] + 1):
        loss = train_sage(
            model, obs_data, obs_feats, obs_labels, criterion, optimizer
        )
        if epoch % conf["eval_interval"] == 0:
            (
                loss_train, 
                score_train,
                loss_val,
                score_val,
                loss_test_tran,
                score_test_tran
            )=eval_on_train_val_test_data_gnn(model, obs_data_eval, obs_feats, obs_labels, criterion, evaluator, obs_idx_train,obs_idx_val,obs_idx_test)                            # Evaluate the inductive part with the full graph
            out, loss_test_ind, score_test_ind = evaluate(
                model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
            )
            print_debug_info_inductive(epoch,loss, loss_train,loss_val,loss_test_tran,loss_test_ind,score_train,score_val,score_test_tran,
            score_test_ind,logger, loss_and_score)

            best_epoch, best_score_val, count,state=early_stop_counter(epoch,model,score_val, best_epoch, best_score_val, count,state)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    out, score_val, score_test_tran, score_test_ind = eval_on_val_test_inductive_gnn(        
        model,
        obs_data_eval,
        obs_feats,
        obs_labels,
        data_eval,
        feats,
        labels,
        criterion,
        evaluator,
        idx_obs,
        obs_idx_val,
        obs_idx_test,
        idx_test_ind,
        logger
    )
    logger.info(
        f"Best valid model at epoch: {best_epoch :3d}, score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind


def distill_run_transductive_sage(
        conf,
        model,
        g,
        feats,
        labels,
        out_t_all,
        distill_indices,
        criterion_l,
        criterion_t,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
    ):
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    idx_l, idx_t, idx_val, idx_test = distill_indices

    data,data_eval= get_train_eval_datasets_sage(g,idx_l,batch_size,conf["fan_out"],conf["num_workers"])

    out_t_all = out_t_all.to(device)

    best_epoch, best_score_val, count,state = 0, 0, 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_l = train_sage(model, data, feats, labels, criterion_l, optimizer,lamb)
        loss_t = train_sage(model, data, feats, out_t_all, criterion_t, optimizer,1-lamb)
        loss = loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            (
                loss_l, 
                score_l,
                loss_val,
                score_val,
                loss_test,
                score_test
            )=eval_on_train_val_test_data_gnn(model, data_eval, feats, labels, criterion_l, evaluator, idx_l,idx_val,idx_test)
            print_debug_info(epoch,loss, loss_l,loss_val,loss_test,score_l,score_val,score_test,logger, loss_and_score)
            best_epoch, best_score_val, count,state=early_stop_counter(epoch,model,score_val, best_epoch, best_score_val, count,state)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    out, _, score_val = evaluate(
        model, data_eval, feats, labels, criterion_l, evaluator, idx_val
    )
    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test


def distill_run_inductive_sage(
    conf,
    model,
    g,
    feats,
    labels,
    out_t_all,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    (
        obs_idx_l,
        obs_idx_t,
        obs_idx_val,
        obs_idx_test,
        idx_obs,
        idx_test_ind,
    ) = distill_indices
    obs_g = g.subgraph(idx_obs)

    obs_data,obs_data_eval= get_train_eval_datasets_sage(obs_g,obs_idx_l,batch_size,conf["fan_out"],conf["num_workers"])
    data,data_eval=get_train_eval_datasets_sage(g,obs_idx_l,batch_size,conf["fan_out"],conf["num_workers"])

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]


    best_epoch, best_score_val, count,state = 0, 0, 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_l = train_sage(
            model, obs_data, obs_feats, obs_labels, criterion_l, optimizer, lamb
        )
        loss_t = train_sage(
            model, obs_data, obs_feats, out_t_all[idx_obs], criterion_t, optimizer, 1-lamb
        )
        loss = loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            (
                loss_train, 
                score_train,
                loss_val,
                score_val,
                loss_test_tran,
                score_test_tran
            )=eval_on_train_val_test_data_gnn(model, obs_data_eval, obs_feats, obs_labels, criterion_l, evaluator, obs_idx_l,obs_idx_val,obs_idx_test)                            # Evaluate the inductive part with the full graph
            out, loss_test_ind, score_test_ind = evaluate(
                model, data_eval, feats, labels, criterion_l, evaluator, idx_test_ind
            )
            print_debug_info_inductive(epoch,loss, loss_train,loss_val,loss_test_tran,loss_test_ind,score_train,score_val,score_test_tran,
            score_test_ind,logger, loss_and_score)

            best_epoch, best_score_val, count,state=early_stop_counter(epoch,model,score_val, best_epoch, best_score_val, count,state)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    out, score_val, score_test_tran, score_test_ind = eval_on_val_test_inductive_gnn(        
        model,
        obs_data_eval,
        obs_feats,
        obs_labels,
        data_eval,
        feats,
        labels,
        criterion_l,
        evaluator,
        idx_obs,
        obs_idx_val,
        obs_idx_test,
        idx_test_ind,
        logger
    )
    logger.info(
        f"Best valid model at epoch: {best_epoch :3d}, score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind
    

