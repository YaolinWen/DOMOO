import os 
import sys 
import wandb 
import numpy as np 
import pandas as pd 
import datetime 
import json 

BASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", ".."
)
sys.path.append(BASE_PATH)

import off_moo_bench as ob
from pymoo.algorithms.moo.nsga2 import NSGA2
from utils import set_seed, get_quantile_solutions
from off_moo_baselines.multi_head.nets import MultiHeadModel
from off_moo_baselines.multi_head.trainer import get_trainer
from off_moo_baselines.multi_head.surrogate_problem import MultiHeadSurrogateProblem
from off_moo_baselines.mo_solver.moea_solver import MOEASolver
from off_moo_baselines.mo_solver.callback import RecordCallback
from off_moo_baselines.data import tkwargs, get_dataloader
from off_moo_bench.collecter import get_operator_dict
from off_moo_bench.task_set import *
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from off_moo_bench.evaluation.metrics import hv, igd
from off_moo_bench.evaluation.plot import plot_y
from scipy.spatial import cKDTree

def igd_off(Y_nds, res_y):
    Y_nds = np.array(Y_nds, dtype=np.float64)
    res_y = np.array(res_y, dtype=np.float64)
    if Y_nds.size == 0 or res_y.size == 0:
        return float('inf')
    mins_per_point = np.min(Y_nds, axis=1)
    t = np.max(mins_per_point)
    Y_true = Y_nds - t
    tree = cKDTree(res_y)
    distances, _ = tree.query(Y_true, k=1)
    igd_value = np.mean(distances)
    return igd_value

def run(config: dict):
    
    if config["task"] in ALLTASKSDICT.keys():
        config["task"] = ALLTASKSDICT[config["task"]]
    
    results_dir = os.path.join(config['results_dir'], 
                               f"{config['model']}-{config['train_mode']}-{config['task']}")
    config["results_dir"] = results_dir 
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f"-ts-{ts.year}-{ts.month}-{ts.day}_{ts.hour}-{ts.minute}-{ts.second}"
    run_name = f"{config['model']}-{config['train_mode']}-seed{config['seed']}-{config['task']}"
    if os.path.exists(results_dir):
        def found(path):
            for x in os.listdir(path):
                if x.startswith('res_y.npy'):
                    return True
            return False
        for f in os.listdir(results_dir):
            if os.path.isdir(os.path.join(results_dir, f)) == False: continue
            if f.startswith(run_name) and found(os.path.join(results_dir, f)):
                print(f"Found existing results: {f}")
                exit()
    
    logging_dir = os.path.join(config['results_dir'], run_name + ts_name)
    os.makedirs(logging_dir, exist_ok=True)

    if config['use_wandb']:
        if 'wandb_api' in config.keys():
            wandb.login(key=config['wandb_api'])

        wandb.init(
            project="Offline-MOO",
            name=run_name + ts_name,
            config=config,
            group=f"{config['model']}-{config['train_mode']}",
            job_type=config['run_type'],
            mode="online",
            dir=os.path.join(config['results_dir'], '..')
        )
    
    with open(os.path.join(logging_dir, "params.json"), "w") as f:
        json.dump(config, f, indent=4)

    set_seed(config['seed'])

    task = ob.make(config['task'])
    
    X = task.x.copy()
    y = task.y.copy()
    
    if config["data_pruning"]:
        X, y = task.get_N_non_dominated_solutions(
            N=int(X.shape[0] * config["data_preserved_ratio"]),
            return_x=True, return_y=True
        )
    
    X_test = task.x_test.copy()
    y_test = task.y_test.copy()
    
    if config['to_logits']:
        assert task.is_discrete 
        task.map_to_logits()
        X = task.to_logits(X)
        X_test = task.to_logits(X_test)
    if config['normalize_xs']:
        task.map_normalize_x()
        X = task.normalize_x(X)
        X_test = task.normalize_x(X_test)
    if config['normalize_ys']:
        task.map_normalize_y()
        y = task.normalize_y(y)
        y_test = task.normalize_y(y_test)
    
    if config['to_logits']:
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
        X_test = X_test.reshape(-1, n_dim * n_classes)
    else:
        data_size, n_dim = tuple(X.shape)
    n_obj = y.shape[1]
        
    model_save_dir = config['model_save_dir']
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_save_path = os.path.join(
        model_save_dir,
        f"{config['model']}-{config['train_mode']}-{config['task']}-{config['seed']}.pt"
    )
    
    model = MultiHeadModel(
        n_dim=n_dim * n_classes if config['to_logits'] else n_dim,
        n_obj=n_obj,
        hidden_size=2048,
        save_path=model_save_path,
    )
    model.set_kwargs(**tkwargs)
    
    trainer_func = get_trainer(config["train_mode"])
    
    trainer = trainer_func(
        forward_model=model, 
        config=config
    )
    
    (
        train_loader,
        val_loader,
        test_loader
    ) = get_dataloader(X, y, X_test, y_test,
                       val_ratio=0.9,
                       batch_size=config["batch_size"])
    
    trainer.launch(
        train_loader,
        val_loader,
        test_loader,
        retrain_model=config["retrain_model"]
    )
    
    surrogate_problem = MultiHeadSurrogateProblem(
        n_var=n_dim * n_classes if config['to_logits'] else n_dim, 
        n_obj=n_obj, model=model
    )
    
    if config["task"] in ScientificDesignSequenceDict.values():
        surrogate_problem.x_to_query_batches = task.problem.task_instance.x_to_query_batches
        surrogate_problem.query_batches_to_x = task.problem.task_instance.query_batches_to_x
        surrogate_problem.candidate_pool = task.problem.task_instance.candidate_pool
        surrogate_problem.op_types = task.problem.task_instance.op_types
    elif config["task"] in MONASSequenceDict.values():
        surrogate_problem.xl = task.problem.xl
        surrogate_problem.xu = task.problem.xu
    
    callback = RecordCallback(
        task=task, surrogate_problem=surrogate_problem,
        config=config, logging_dir=logging_dir, iters_to_record=1
    )
    
    genetic_operators = get_operator_dict(config)
    
    solver = MOEASolver(
        n_gen=config["solver_n_gen"],
        pop_init_method=config["solver_init_method"],
        batch_size=config["num_solutions"],
        pop_size=config["num_solutions"],
        algo=NSGA2,
        callback=callback if config["record_hist"] else None,
        eliminate_duplicates=True,
        **genetic_operators,
    )
    
    res = solver.solve(surrogate_problem, X=X, Y=y)
    
    res_x = res["x"]
    if config['to_logits']:
        res_x = res_x.reshape(-1, n_dim, n_classes)
    if config['normalize_xs']:
        task.map_denormalize_x()
        res_x = task.denormalize_x(res_x)
    if config['to_logits']:
        task.map_to_integers()
        res_x = task.to_integers(res_x)
    
    res_y = task.predict(res_x)
    # I noticed that there's a weird bug for task DTLZ2, the shape of res_x
    # is (batch_size, n_dim), but the shape of res_y is (n_obj, batch_size)
    # Need to fix this issue and understand why this happens
    # Simply transpose the res_y
    if res_y.shape[0] != res_x.shape[0]:
        res_y = res_y.T
    visible_masks = np.ones(len(res_y))
    visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
    visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
    res_x = res_x[np.where(visible_masks == 1)[0]]
    res_y = res_y[np.where(visible_masks == 1)[0]]
    
    res_y_75_percent = get_quantile_solutions(res_y, 0.75)
    res_y_50_percent = get_quantile_solutions(res_y, 0.50)
    
    nadir_point = task.nadir_point
    if config['normalize_ys']:
        res_y = task.normalize_y(res_y)
        nadir_point = task.normalize_y(nadir_point)
        res_y_50_percent = task.normalize_y(res_y_50_percent)
        res_y_75_percent = task.normalize_y(res_y_75_percent)
        
    _, d_best = task.get_N_non_dominated_solutions(
        N=config["num_solutions"], 
        return_x=False, return_y=True
    )
    
    np.save(file=os.path.join(logging_dir, "res_x.npy"), arr=res_x)
    np.save(file=os.path.join(logging_dir, "res_y.npy"), arr=res_y)
    plot_y(res_y, save_dir=logging_dir, config=config,
           nadir_point=nadir_point, d_best=d_best)
        
    d_best_hv = hv(nadir_point, d_best, config['task'])
    hv_value = hv(nadir_point, res_y, config['task'])
    hv_value_50_percentile = hv(nadir_point, res_y_50_percent, config['task'])
    hv_value_75_percentile = hv(nadir_point, res_y_75_percent, config['task'])
    
    print(f"Hypervolume (100th): {hv_value:4f}")
    print(f"Hypervolume (75th): {hv_value_75_percentile:4f}")
    print(f"Hypervolume (50th): {hv_value_50_percentile:4f}")
    print(f"Hypervolume (D(best)): {d_best_hv:4f}")
    
    hv_results = {
        "hypervolume/D(best)": d_best_hv,
        "hypervolume/100th": hv_value, 
        "hypervolume/75th": hv_value_75_percentile,
        "hypervolume/50th": hv_value_50_percentile,
        "evaluation_step": 1,
    }
    
    df = pd.DataFrame([hv_results])
    filename = os.path.join(logging_dir, "hv_results.csv")
    df.to_csv(filename, index=False)

    nds = NonDominatedSorting()
    idx_nds = nds.do(y)  
    X_nds = X[idx_nds[0]] 
    Y_nds = y[idx_nds[0]] 
    
    d_best_igd_off = igd_off(Y_nds, d_best)
    igd_off_value = igd_off(Y_nds, res_y)
    igd_off_value_50_percentile = igd_off(Y_nds, res_y_50_percent)
    igd_off_value_75_percentile = igd_off(Y_nds, res_y_75_percent)
            
    print(f"IGD-offline (100th): {igd_off_value:4f}")
    print(f"IGD-offline (75th): {igd_off_value_75_percentile:4f}")
    print(f"IGD-offline (50th): {igd_off_value_50_percentile:4f}")
    print(f"IGD-offline (D(best)): {d_best_igd_off:4f}")
            
    igd_off_results = {
                "igd-offline/D(best)": d_best_igd_off,
                "igd-offline/100th": igd_off_value,
                "igd-offline/75th": igd_off_value_75_percentile,
                "igd-offline/50th": igd_off_value_50_percentile,
                "evaluation_step": 1,
    }
    df = pd.DataFrame([igd_off_results])
    filename = os.path.join(logging_dir, "igd_off_results.csv")
    df.to_csv(filename, index=False)
    
    if config["use_wandb"]:
        wandb.log(hv_results)
        
if __name__ == "__main__":
    from utils import process_args 
    config = process_args(return_dict=True)
    
    results_dir = os.path.join(BASE_PATH, "results")
    model_save_dir = os.path.join(BASE_PATH, "model")
    
    config["results_dir"] = results_dir
    config["model_save_dir"] = model_save_dir
    
    run(config)