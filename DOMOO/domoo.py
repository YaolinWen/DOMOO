import sys, os
import datetime 
import json 
import yaml
from tqdm import tqdm
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from pymoo.core.problem import Problem
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from offline_moo.off_moo_bench.evaluation.metrics import hv, igd
from offline_moo.off_moo_baselines.data import tkwargs, get_dataloader
from offline_moo.off_moo_baselines.multiple_models.trainer import get_trainer
from offline_moo.off_moo_baselines.multiple_models.nets import MultipleModels
from energy_model import RiskSuppressionFactor, sample_langevin, DualHeadSurogateTrainer, DualHeadSurogateModel
from exper import get_parser
from utils import *

def run(args):
    task_name = ALLTASKSDICT[args.task]
    results_dir = os.path.join(args.results_dir, 
                                f"{args.model}-{args.train_mode}-{task_name}")
    args.results_dir = results_dir 

    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f"-ts-{ts.year}-{ts.month}-{ts.day}_{ts.hour}-{ts.minute}-{ts.second}"
    run_name = f"{args.model}-{args.train_mode}-seed{args.seed}-{task_name}"
    
    if os.path.exists(results_dir) and args.skip_existing:
        def found(path):
            for x in os.listdir(path):
                if x.startswith('igd_off_results.csv'):
                    return True
            return False
        for f in os.listdir(results_dir):
            if os.path.isdir(os.path.join(results_dir, f)) == False: continue
            if f.startswith(run_name) and found(os.path.join(results_dir, f)):
                print(f"Found existing results: {f}")
                exit()
    res_x = load_other_result(args)

    logging_dir = os.path.join(args.results_dir, run_name + ts_name)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.energy_save_dir, exist_ok=True)
    task, X, y, X_test, y_test, n_obj, n_classes = define_task(args)
    n_dim = len(X[0])

    model = MultipleModels(
        n_dim=n_dim,
        n_obj=n_obj,
        train_mode=args.train_mode,
        hidden_size=[2048, 2048],
        save_dir=args.model_save_dir,
        save_prefix=f"{args.model}-{args.train_mode}-{args.task}-{args.seed}"
    )
    model.set_kwargs(**tkwargs)

    trainer_func = get_trainer(args.train_mode)
    init_m = 0.02 * np.sqrt(np.prod(X.shape[1:]))
    nds = NonDominatedSorting()
    idx_nds = nds.do(y)  
    X_nds = X[idx_nds[0]] 
    Y_nds = y[idx_nds[0]] 


    
    init_x = X_nds
    if len(init_x) < 256:
        XX = np.delete(X, idx_nds[0], axis=0)
        YY = np.delete(y, idx_nds[0], axis=0)
        idx_nds = nds.do(YY)  
        X_nds_2 = XX[idx_nds[0]] 
        Y_nds_2 = YY[idx_nds[0]] 
        init_x = np.concatenate((X_nds, X_nds_2), axis=0)
    if len(init_x) > 256:
        random_indices = np.random.choice(init_x.shape[0], size=256, replace=False)
        init_x = init_x[random_indices]
    init_x = torch.tensor(init_x).cuda().to(torch.float)

    dhs_models = []
    uc_li = []
    for which_obj in range(n_obj):
        dhs_model = DualHeadSurogateModel(
            n_dim if which_obj == 0 else np.prod(X.shape[1:]),  
            [512, 512], 
            save_dir=args.energy_save_dir, 
            which_obj=which_obj,
            save_prefix=f"{args.model}-{args.train_mode}-{args.task}-{args.seed}-energy-{which_obj}"
        ).cuda()
        print(args.energy_save_dir)
        print(f"{args.model}-{args.train_mode}-{args.task}-{args.seed}-energy-{which_obj}")
        trainer_dhs = DualHeadSurogateTrainer(
            dhs_model,
            model.obj2model[which_obj], 
            dhs_model_energy_opt=torch.optim.Adam,
            surrogate_lr=0.001,
            init_m=init_m,
            ldk=args.Ld_K
        )
        dhs_models.append(dhs_model)

        y0 = y[:, which_obj].copy().reshape(-1, 1)
        
        y0_test = y_test[:, which_obj].copy().reshape(-1, 1)
        
        args.which_obj = which_obj
        args.input_shape = X[0].shape
        
        indexs = np.argsort(y0.squeeze())
        index = indexs[:args.num_solutions]
        args.best_x = torch.from_numpy(deepcopy(X[index])).to(**tkwargs)
        args.best_y = torch.from_numpy(deepcopy(y0[index])).to(**tkwargs)
        
        args.x = torch.from_numpy(deepcopy(X)).to(**tkwargs)
        args.y = torch.from_numpy(deepcopy(y0)).to(**tkwargs)
        trainer = trainer_func(
            model=list(model.obj2model.values())[which_obj], 
            config=args
        )
        
        (
            train_loader,
            val_loader,
            test_loader
        ) = get_dataloader(X, y0, X_test, y0_test,
                        val_ratio=0.9,
                        batch_size=args.batch_size)
        
        trainer.launch(
            train_loader,
            val_loader,
            test_loader,
            retrain_model=args.retrain_model
        )
        trainer_dhs.launch(train_loader, args.train_epoch, args.retrain_model)
        energy_min = dhs_model(init_x).mean().detach().cpu().numpy()
        energy_max = dhs_model(sample_langevin(init_x, model.obj2model[which_obj],
                                                    stepsize=init_m,
                                                    n_steps=args.Ld_K_max,
                                                    noise=False
                                                    )).mean().detach().cpu().numpy()
        uc = RiskSuppressionFactor(energy_min, energy_max, init_m = init_m)
        uc_li.append(uc)
        
    res_y = model(torch.tensor(res_x,dtype=torch.float32).to(args.device)).detach().cpu().numpy()
    
    data_x, data_y = X, y 
    z = torch.zeros(n_obj).to(args.device)
    z = torch.min(torch.cat((z.reshape(1, n_obj), torch.from_numpy(y).to(args.device) - 0.1)), axis=0).values.data
    nds = NonDominatedSorting()
    idx_nds = nds.do(data_y)  
    preferences = np.array([generate_preference(tmp_y, z.detach().cpu().numpy()) for tmp_y in data_y[idx_nds[0]]])

    psmodel_pretrain_nepoch=args.psmodel_pretrain_nepoch
    t_steps = args.t_steps
    n_steps = args.n_steps
    risk_ratio = args.risk_ratio

    psmodel = psmodel_init(
        args,
        data_x.shape[1], 
        n_obj, 
        data_x[idx_nds[0]] ,
        data_y[idx_nds[0]], 
        preferences,z, 
        nepoch=psmodel_pretrain_nepoch
    )

    psmodel_train(args, psmodel, dhs_models, model, uc_li, n_obj,z, t_steps=t_steps, n_steps=n_steps, risk_ratio=risk_ratio)
    
    alpha = np.ones(n_obj)
    preference_vectors = np.random.dirichlet(alpha, size=args.n_candidate)
    preference_vectors = torch.tensor(preference_vectors, dtype=torch.float32, device=args.device) + 1e-4
    X_candidate = psmodel(preference_vectors)
    X_candidate_tch = psmodel(preference_vectors).to(torch.float32)
    X_candidate_np = X_candidate_tch.detach().cpu().numpy()
    Y_candidate = model(X_candidate_tch)

    Y_p = Y_nds
    YY_candidate = Y_candidate.detach().cpu().numpy()
    X_candidates_np = X_candidate_np
    X_candidates_np

    np.savetxt(os.path.join(logging_dir, f"YY_candidate.txt"), YY_candidate)
    np.savetxt(os.path.join(logging_dir, f"X_candidate_np.txt"), X_candidate_np)


    
    if args.no_psl:
        X_candidates_np = res_x
        YY_candidate = res_y
    elif args.no_surrogate:
        X_candidates_np = X_candidate_np[:512]
        YY_candidate = YY_candidate[:512]
    else:
        X_candidates_np = np.vstack([X_candidate_np[:256], res_x])
        YY_candidate = np.vstack([YY_candidate[:256], res_y])

    nadir_point = task.nadir_point
    nadir_point = task.normalize_y(nadir_point, normalization_method="min-max")
    _, d_best = task.get_N_non_dominated_solutions(N=256, return_x=False, return_y=True)
    d_best = task.normalize_y(d_best, normalization_method="min-max")

    best_subset_list = []
    if not args.no_igdoff:
        best_subset_list = select_igdoff(128, best_subset_list, X_candidates_np, YY_candidate, Y_nds)
    best_subset_list = select_hv(args.n_sample - len(best_subset_list),best_subset_list, X_candidates_np, YY_candidate, nadir_point=nadir_point, taskname=task_name)
    best_subset_list = select_random(args.n_sample - len(best_subset_list), best_subset_list, X_candidates_np, YY_candidate, do_nds=True)
    best_subset_list = np.array(best_subset_list).T[0]
    evaluate(task, X_candidates_np, best_subset_list, args, n_dim, n_classes, Y_nds, logging_dir, final=True)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = get_parser()
    set_seed(args.seed)
    run(args)