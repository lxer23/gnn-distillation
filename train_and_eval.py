import numpy as np
import copy
import torch
import dgl
from utils import set_seed
from train_and_eval_utils.mlp import (
    run_transductive_mlp,
    run_inductive_mlp,
    distill_run_transductive_mlp,
    distill_run_inductive_mlp
)
from train_and_eval_utils.gnn import run_transductive_gnn, run_inductive_gnn, distill_run_transductive_gnn,distill_run_inductive_gnn
from train_and_eval_utils.sage import run_transductive_sage, run_inductive_sage,distill_run_transductive_sage,distill_run_inductive_sage
from train_and_eval_utils.utils import print_debug_info, early_stop_counter, print_debug_info_inductive

"""
2. Run teacher
"""
def run_transductive(
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
    """
    Train and eval under the transductive setting.
    The train/valid/test split is specified by `indices`.
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]

    idx_train, idx_val, idx_test = indices

    feats = feats.to(device)
    labels = labels.to(device)

    if "SAGE" in model.model_name:
        return run_transductive_sage(
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
        )
    elif "MLP" in model.model_name:
        return run_transductive_mlp(
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
        )
    else:
        return run_transductive_gnn(
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
        )


def run_inductive(
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
    """
    Train and eval under the inductive setting.
    The train/valid/test split is specified by `indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_g = g.subgraph(idx_obs)

    if "SAGE" in model.model_name:
        return run_inductive_sage(    
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
        )

    elif "MLP" in model.model_name:
        return run_inductive_mlp(    
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
        )

    else:
        return run_inductive_gnn(    
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
        )


"""
3. Distill
"""


def distill_run_transductive(
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
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """

    if "SAGE" in model.model_name:
        return distill_run_transductive_sage(
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
        )

    elif "MLP" in model.model_name:
        return distill_run_transductive_mlp(
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
        )

    else:
        return distill_run_transductive_gnn(
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
        )


def distill_run_inductive(
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
    if "SAGE" in model.model_name:
        return distill_run_inductive_sage(
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
        )

    elif "MLP" in model.model_name:
        return distill_run_inductive_mlp(
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
        )

    else:
        return distill_run_inductive_gnn(
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
        )
