import numpy as np
import copy
import torch
import dgl
from utils import set_seed
from train_and_eval_utils.utils import print_debug_info, early_stop_counter, print_debug_info_inductive
"""
1. Train and eval
"""


def train_mini_batch(model, feats, labels, batch_size, criterion, optimizer, lamb=1):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        # No graph needed for the forward function
        logits = model(None, feats[idx_batch[i]])
        out = logits.log_softmax(dim=1)

        loss = criterion(out, labels[idx_batch[i]])
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


def evaluate_mini_batch(
    model, feats, labels, criterion, batch_size, evaluator, idx_eval=None
):
    """
    Evaluate MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    Return:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """

    model.eval()
    with torch.no_grad():
        num_batches = int(np.ceil(len(feats) / batch_size))
        out_list = []
        for i in range(num_batches):
            logits = model.inference(None, feats[batch_size * i : batch_size * (i + 1)])
            out = logits.log_softmax(dim=1)
            out_list += [out.detach()]

        out_all = torch.cat(out_list)

        if idx_eval is None:
            loss = criterion(out_all, labels)
            score = evaluator(out_all, labels)
        else:
            loss = criterion(out_all[idx_eval], labels[idx_eval])
            score = evaluator(out_all[idx_eval], labels[idx_eval])

    return out_all, loss.item(), score

def eval_on_train_val_test_data_mlp(model, feats_train,labels_train,feats_val, labels_val,feats_test, labels_test,criterion,batch_size,evaluator):
    _, loss_train, score_train = evaluate_mini_batch(
        model, feats_train, labels_train, criterion, batch_size, evaluator
    )
    _, loss_val, score_val = evaluate_mini_batch(
        model, feats_val, labels_val, criterion, batch_size, evaluator
    )
    _, loss_test, score_test = evaluate_mini_batch(
        model, feats_test, labels_test, criterion, batch_size, evaluator
    )
    return loss_train, score_train,loss_val,score_val,loss_test,score_test



def run_transductive_mlp(
    conf,
    model,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    batch_size = conf["batch_size"]
    idx_train, idx_val, idx_test = indices

    feats_train, labels_train = feats[idx_train], labels[idx_train]
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]

    best_epoch, best_score_val, count, state = 0, 0, 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, conf["max_epoch"] + 1):
        loss = train_mini_batch(
            model, feats_train, labels_train, batch_size, criterion, optimizer
        )
        if epoch % conf["eval_interval"] == 0:
            (
                loss_train, 
                score_train,
                loss_val,
                score_val,
                loss_test,
                score_test
            )=eval_on_train_val_test_data_mlp(model, feats_train,labels_train,feats_val, labels_val,feats_test, labels_test,criterion,batch_size,evaluator)
            print_debug_info(epoch,loss, loss_train,loss_val,loss_test,score_train,score_val,score_test,logger, loss_and_score)
            best_epoch, best_score_val, count,state= early_stop_counter(epoch,model,score_val, best_epoch, best_score_val, count,state)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break
    model.load_state_dict(state)
    out, _, score_val = evaluate_mini_batch(
        model, feats, labels, criterion, batch_size, evaluator, idx_val
    )
    
    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test

def step_eval_ind_mlp(epoch,model,loss,feats_labels,criterion,batch_size,evaluator,save_info,count):
    feats_train=feats_labels["feats_train"]
    labels_train=feats_labels["labels_train"]
    feats_val=feats_labels["feats_val"]
    labels_val=feats_labels["labels_val"]
    feats_test_tran=feats_labels["feats_test_tran"]
    labels_test_tran=feats_labels["labels_test_tran"]
    feats_test_ind=feats_labels["feats_test_ind"]
    labels_test_ind=feats_labels["labels_test_ind"]
    logger,loss_and_score= save_info
    (
        loss_train, 
        score_train,
        loss_val,
        score_val,
        loss_test_tran,
        score_test_tran
    )=eval_on_train_val_test_data_mlp(model, feats_train,labels_train,feats_val, labels_val,feats_test_tran, labels_test_tran,criterion,batch_size,evaluator)
    _, loss_test_ind, score_test_ind = evaluate_mini_batch(
        model,
        feats_test_ind,
        labels_test_ind,
        criterion,
        batch_size,
        evaluator,
    )
    print_debug_info_inductive(epoch,loss, loss_train,loss_val,loss_test_tran,loss_test_ind,score_train,score_val,score_test_tran,score_test_ind,logger,loss_and_score)
    return score_val


def run_inductive_mlp(    
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
    feats_train, labels_train = obs_feats[obs_idx_train], obs_labels[obs_idx_train]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = (
        obs_feats[obs_idx_test],
        obs_labels[obs_idx_test],
    )
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    feats_labels={
        "feats_train": feats_train,
        "labels_train": labels_train,
        "feats_val": feats_val,
        "labels_val": labels_val,
        "feats_test_tran": feats_test_tran,
        "labels_test_tran": labels_test_tran,
        "feats_test_ind" :feats_test_ind,
        "labels_test_ind": labels_test_ind
    }
    save_info=(logger,loss_and_score)

    best_epoch, best_score_val, count, state = 0, 0, 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, conf["max_epoch"] + 1):
        loss = train_mini_batch(
            model, feats_train, labels_train, batch_size, criterion, optimizer
        )
        if epoch % conf["eval_interval"] == 0:
            score_val=step_eval_ind_mlp(epoch,model,loss,feats_labels,criterion,batch_size,evaluator,save_info,count)
            best_epoch, best_score_val, count, state=early_stop_counter(epoch,model,score_val, best_epoch, best_score_val, count,state)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    obs_out, _, score_val = evaluate_mini_batch(
        model, obs_feats, obs_labels, criterion, batch_size, evaluator, obs_idx_val
    )
    out, _, score_test_ind = evaluate_mini_batch(
        model, feats, labels, criterion, batch_size, evaluator, idx_test_ind
    )
    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out[idx_obs] = obs_out
    logger.info(
        f"Best valid model at epoch: {best_epoch :3d}, score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind

def distill_run_transductive_mlp(
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

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)

    feats_l, labels_l = feats[idx_l], labels[idx_l]
    feats_t, out_t = feats[idx_t], out_t_all[idx_t]
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]

    best_epoch, best_score_val, count,state = 0, 0, 0,copy.deepcopy(model.state_dict())
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_l = train_mini_batch(
            model, feats_l, labels_l, batch_size, criterion_l, optimizer, lamb
        )
        loss_t = train_mini_batch(
            model, feats_t, out_t, batch_size, criterion_t, optimizer, 1 - lamb
        )
        loss = loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            (
                loss_l, 
                score_l,
                loss_val,
                score_val,
                loss_test,
                score_test
            )=eval_on_train_val_test_data_mlp(model, feats_l,labels_l,feats_val, labels_val,feats_test, labels_test,criterion_l,batch_size,evaluator)

            print_debug_info(epoch,loss, loss_l,loss_val,loss_test,score_l,score_val,score_test,logger, loss_and_score)

            best_epoch, best_score_val, count,state=early_stop_counter(epoch,model,score_val, best_epoch, best_score_val, count,state)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    out, _, score_val = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_val
    )
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test = evaluator(out[idx_test], labels_test)

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test


def distill_run_inductive_mlp(
    conf,
    model,
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

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_out_t = out_t_all[idx_obs]

    feats_l, labels_l = obs_feats[obs_idx_l], obs_labels[obs_idx_l]
    feats_t, out_t = obs_feats[obs_idx_t], obs_out_t[obs_idx_t]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = (
        obs_feats[obs_idx_test],
        obs_labels[obs_idx_test],
    )
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    best_epoch, best_score_val, count,state = 0, 0, 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_l = train_mini_batch(
            model, feats_l, labels_l, batch_size, criterion_l, optimizer, lamb
        )
        loss_t = train_mini_batch(
            model, feats_t, out_t, batch_size, criterion_t, optimizer, 1 - lamb
        )
        loss = loss_l + loss_t

        if epoch % conf["eval_interval"] == 0:
            (
                loss_l, 
                score_l,
                loss_val,
                score_val,
                loss_test_tran,
                score_test_tran
            )=eval_on_train_val_test_data_mlp(model, feats_l,labels_l,feats_val, labels_val,feats_test_tran, labels_test_tran,criterion_l,batch_size,evaluator)
            _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                model,
                feats_test_ind,
                labels_test_ind,
                criterion_l,
                batch_size,
                evaluator,
            )
            print_debug_info_inductive(epoch,loss, loss_l,loss_val,loss_test_tran,loss_test_ind,score_l,score_val,score_test_tran,score_test_ind,logger, loss_and_score)
            best_epoch, best_score_val, count,state=early_stop_counter(epoch,model,score_val, best_epoch, best_score_val, count,state)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    obs_out, _, score_val = evaluate_mini_batch(
        model, obs_feats, obs_labels, criterion_l, batch_size, evaluator, obs_idx_val
    )
    out, _, score_test_ind = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_test_ind
    )
    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out[idx_obs] = obs_out
    logger.info(
        f"Best valid model at epoch: {best_epoch :3d}, score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind
    
