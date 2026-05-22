import random, os, torch
import numpy as np
from offline_moo.off_moo_bench.evaluation.metrics import hv
from typing import List, Optional
import torch.autograd as autograd
import offline_moo.off_moo_bench as ob
import pandas as pd 
from offline_moo.off_moo_bench.evaluation.metrics import hv, igd
from offline_moo.off_moo_bench.evaluation.plot import plot_y
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

tkwargs = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dtype': torch.float32
}
import time


def plot_pareto_front(objectives):
    num_objectives = 2
    
    if num_objectives == 2:  # 仅支持2D目标
        plt.figure(figsize=(8, 6))
        plt.scatter(objectives[:, 0], objectives[:, 1], c='blue', label='Solutions')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Pareto Front Visualization')
        plt.grid(True)
        plt.legend()
        plt.savefig("pareto_front.png", format='png')
        plt.close()

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

# We don't use permutation tasks in our experiments
SyntheticFunctionDict = {
    "zdt1": "ZDT1-Exact-v0",
    "zdt2": "ZDT2-Exact-v0",
    "zdt3": "ZDT3-Exact-v0",
    "zdt4": "ZDT4-Exact-v0",
    "zdt6": "ZDT6-Exact-v0",
    "omnitest": "OmniTest-Exact-v0",
    "vlmop1": "VLMOP1-Exact-v0",
    "vlmop2": "VLMOP2-Exact-v0",
    "vlmop3": "VLMOP3-Exact-v0",
    "dtlz1": "DTLZ1-Exact-v0",
    "dtlz2": "DTLZ2-Exact-v0",
    "dtlz3": "DTLZ3-Exact-v0",
    "dtlz4": "DTLZ4-Exact-v0",
    "dtlz5": "DTLZ5-Exact-v0",
    "dtlz6": "DTLZ6-Exact-v0",
    "dtlz7": "DTLZ7-Exact-v0",
}

MONASSequenceDict = {
    "c10mop1": "C10MOP1-Exact-v0",
    "c10mop2": "C10MOP2-Exact-v0",
    "c10mop3": "C10MOP3-Exact-v0",
    "c10mop4": "C10MOP4-Exact-v0",
    "c10mop5": "C10MOP5-Exact-v0",
    "c10mop6": "C10MOP6-Exact-v0",
    "c10mop7": "C10MOP7-Exact-v0",
    "c10mop8": "C10MOP8-Exact-v0",
    "c10mop9": "C10MOP9-Exact-v0",
    "in1kmop1": "IN1KMOP1-Exact-v0",
    "in1kmop2": "IN1KMOP2-Exact-v0",
    "in1kmop3": "IN1KMOP3-Exact-v0",
    "in1kmop4": "IN1KMOP4-Exact-v0",
    "in1kmop5": "IN1KMOP5-Exact-v0",
    "in1kmop6": "IN1KMOP6-Exact-v0",
    "in1kmop7": "IN1KMOP7-Exact-v0",
    "in1kmop8": "IN1KMOP8-Exact-v0",
    "in1kmop9": "IN1KMOP9-Exact-v0",
}

MONASLogitsDict = {
    "nb201_test": "NASBench201Test-Exact-v0",
}

MOCOContinuousDict = {"portfolio": "Portfolio-Exact-v0"}

MORLDict = {
    "mo_swimmer_v2": "MOSwimmerV2-Exact-v0",
    "mo_hopper_v2": "MOHopperV2-Exact-v0",
}

ScientificDesignContinuousDict = {
    "molecule": "Molecule-Exact-v0",
}

ScientificDesignSequenceDict = {
    "regex": "Regex-Exact-v0",
    "zinc": "ZINC-Exact-v0",
    "rfp": "RFP-Exact-v0",
}

RESuiteDict = {
    "re21": "RE21-Exact-v0",
    "re22": "RE22-Exact-v0",
    "re23": "RE23-Exact-v0",
    "re24": "RE24-Exact-v0",
    "re25": "RE25-Exact-v0",
    "re31": "RE31-Exact-v0",
    "re32": "RE32-Exact-v0",
    "re33": "RE33-Exact-v0",
    "re34": "RE34-Exact-v0",
    "re35": "RE35-Exact-v0",
    "re36": "RE36-Exact-v0",
    "re37": "RE37-Exact-v0",
    "re41": "RE41-Exact-v0",
    "re42": "RE42-Exact-v0",
    "re61": "RE61-Exact-v0",
}

SyntheticFunction = list(SyntheticFunctionDict.values())
MONASSequence = list(MONASSequenceDict.values())
MONASLogits = list(MONASLogitsDict.values())
MOCOContinuous = list(MOCOContinuousDict.values())
MORL = list(MORLDict.values())
ScientificDesignContinuous = list(ScientificDesignContinuousDict.values())
ScientificDesignSequence = list(ScientificDesignSequenceDict.values())
RESuite = list(RESuiteDict.values())

MONAS = MONASSequence + MONASLogits
MOCO = MOCOContinuous
ScientificDesign = ScientificDesignContinuous + ScientificDesignSequence

ALLTASKS = SyntheticFunction + MONAS + MOCO + MORL + ScientificDesign + RESuite
ALLTASKSDICT = {
    **SyntheticFunctionDict,
    **MONASSequenceDict,
    **MONASLogitsDict,
    **MOCOContinuousDict,
    **MORLDict,
    **ScientificDesignContinuousDict,
    **ScientificDesignSequenceDict,
    **RESuiteDict,
}

CONTINUOUSTASKS = (
    SyntheticFunction
    + MONASLogits
    + MOCOContinuous
    + MORL
    + ScientificDesignContinuous
    + RESuite
)
SEQUENCETASKS = MONASSequence + ScientificDesignSequence

# Get all keys in the dictionary
all_task_names = list(ALLTASKSDICT.keys())

START_TIMESTAMP = time.time()
def define_task(args):
    task_name = ALLTASKSDICT[args.task]
    print(f"Task:{task_name}")
    set_seed(args.seed)
    task = ob.make(task_name)
    X = task.x.copy()
    y = task.y.copy()
    
    try:
        if args.data_pruning:
            X, y = task.get_N_non_dominated_solutions(
                N=int(X.shape[0] * args.data_preserved_ratio),
                return_x=True, return_y=True
            )
    except:
        print("data_pruning failed")
        pass
    
    X_test = task.x_test.copy()
    y_test = task.y_test.copy()
    n_obj = len(y[0])
    n_classes = 1
    if task.is_discrete:
        X = task.to_logits(X)
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
        X_test = task.to_logits(X_test)
        X_test = X_test.reshape(-1, n_dim * n_classes)
    if task.is_sequence:
        X = task.to_logits(X)
        X_test = task.to_logits(X_test)
    
    if args.normalization:
        X = task.normalize_x(X, normalization_method="min-max")
        X_test = task.normalize_x(X_test, normalization_method="min-max")
    y = task.normalize_y(y, normalization_method="min-max")
    y_test = task.normalize_y(y_test, normalization_method="min-max")
    return task, X, y, X_test, y_test, n_obj, n_classes

def load_other_result(args):
    other_root = args.multiple_model_results
    other_dir_name = f'MultipleModels-Vallina-{ALLTASKSDICT[args.task]}'
    dirs = []
    for dir in os.listdir(os.path.join(other_root, other_dir_name)):
        if f'seed{args.seed}' in dir :
            dirs.append(dir)
    dirs.sort()
    other_dir = os.path.join(other_root, other_dir_name, dirs[-1] )
    print(other_dir)

    res_x = np.load(os.path.join(other_dir, 'res_x.npy'))
    if args.task in ['regex', 'rfp', 'zinc']:
        if args.task in ['regex', 'rfp', 'zinc']:
            global weird_solutions
            weird_solutions = res_x[np.any(res_x<0,axis=1)]
        res_x = res_x[np.all(res_x >= 0, axis=1)]
        
    task_name = ALLTASKSDICT[args.task]
    print(f"Task:{task_name}")
    set_seed(args.seed)
    task = ob.make(task_name)
    X = task.x.copy()
    y = task.y.copy()
    
    try:
        if args.data_pruning:
            X, y = task.get_N_non_dominated_solutions(
                N=int(X.shape[0] * args.data_preserved_ratio),
                return_x=True, return_y=True
            )
    except:
        print("data_pruning failed")
        pass
    
    X_test = task.x_test.copy()
    y_test = task.y_test.copy()
    n_obj = len(y[0])
    n_classes = 1
    if task.is_discrete:
        X = task.to_logits(X)
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
        X_test = task.to_logits(X_test)
        X_test = X_test.reshape(-1, n_dim * n_classes) # wyl
        res_x = task.to_logits(res_x)
        res_x = res_x.reshape(-1, n_dim * n_classes)
    if task.is_sequence:
        X = task.to_logits(X)
        X_test = task.to_logits(X_test)
        res_x = task.to_logits(res_x)
    
    
    if args.normalization:
        X = task.normalize_x(X, normalization_method="min-max")
        X_test = task.normalize_x(X_test, normalization_method="min-max")
        res_x = task.normalize_x(res_x, normalization_method="min-max")

    return res_x

def load_pareto(model, x_np, pref_vec, z, args):
    output = model(x_np)
    num_outputs = output.shape[1]
    value_grads = []
    for i in range(num_outputs):
        value = output[:, i]
        if i < num_outputs - 1:
            value_grad = autograd.grad(value.sum(), x_np, retain_graph=True)[0]
        else:
            value_grad = autograd.grad(value.sum(), x_np)[0]
        value_grads.append(value_grad)
    value_grad = torch.stack(value_grads, dim=1)
        
    return value_grad



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

weird_solutions = []
def evaluate(task, X_candidate_np, best_subset_list, args, dim, n_classes, Y_nds, logging_dir, final=False):
    X_candidate = torch.tensor(X_candidate_np).to(args.device)
    X_new = X_candidate[best_subset_list]
    if isinstance(X_new, torch.Tensor):
        X_new = X_new.detach().cpu().numpy()
    try:
        X_new  = X_new.detach().cpu().numpy()
    except:
        pass
    res_x = X_new
    if args.normalization:
        X_new = task.denormalize_x(X_new, normalization_method="min-max")
        res_x = X_new
    if task.is_discrete:
        res_x = res_x.reshape(-1, dim//n_classes, n_classes)
        res_x = task.to_integers(res_x)
    if task.is_sequence:
        res_x = task.to_integers(res_x)
    
    # try:
    global weird_solutions
    if len(weird_solutions)>0 and final:
        weird_solutions = np.array(weird_solutions)[:128]
        res_x = np.vstack([res_x[:-len(weird_solutions)], weird_solutions])

    res_y = task.predict(np.array(res_x))
    # bug-fixer from paretoflow, dtlz-2
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
    nadir_point = task.normalize_y(nadir_point, normalization_method="min-max")
    res_y = task.normalize_y(res_y, normalization_method="min-max")
    res_y_50_percent = task.normalize_y(res_y_50_percent, normalization_method="min-max")
    res_y_75_percent = task.normalize_y(res_y_75_percent, normalization_method="min-max")
        
    _, d_best = task.get_N_non_dominated_solutions(N=256, return_x=False, return_y=True)
    d_best = task.normalize_y(d_best, normalization_method="min-max")
    np.save(file=os.path.join(logging_dir, "res_x.npy"), arr=res_x)
    np.save(file=os.path.join(logging_dir, "res_y.npy"), arr=res_y)
    
    task_name = args.task
    if task_name in ALLTASKSDICT.keys():
        task_name = ALLTASKSDICT[task_name]
    plot_pareto_front(res_y)
    try:
        config = {
            "task": task_name,
        }
        plot_y(
            res_y,
            save_dir=logging_dir,
            config=config,
            nadir_point=nadir_point,
            d_best=d_best,
        )
    except:
        print("[error] plot_y error")

    d_best_hv = hv(nadir_point, d_best, task_name)
    hv_value = hv(nadir_point, res_y, task_name)
    hv_value_50_percentile = hv(nadir_point, res_y_50_percent, task_name)
    hv_value_75_percentile = hv(nadir_point, res_y_75_percent, task_name)
    
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

    global START_TIMESTAMP
    start_time = START_TIMESTAMP
    print('time elapsed:', time.time() - start_time, ' seconds')

def get_quantile_solutions(Y_all: np.ndarray, quantile) -> np.ndarray:
    assert 0 < quantile < 1
    n = len(Y_all)
    n_remove = int(n * (1-quantile))
    indices_to_remove = get_N_nondominated_index(Y_all, n_remove)
    indices_to_keep = np.ones(n)
    indices_to_keep[indices_to_remove] = 0
    return Y_all[np.where(indices_to_keep == 1)[0]]

def _get_fronts(Y_all):
    global now_fronts
    if now_fronts is not None:
        return now_fronts
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    fronts = NonDominatedSorting().do(Y_all, return_rank=True)[0]
    now_fronts = fronts 
    return fronts

def get_N_nondominated_index(Y_all, num_ret, is_all_data=False) -> List[int]:
    if is_all_data:
        fronts = _get_fronts(Y_all)
    else:
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        fronts = NonDominatedSorting().do(Y_all, return_rank=True, n_stop_if_ranked=num_ret)[0]
    indices_cnt = 0
    indices_select = []
    for front in fronts:
        if indices_cnt + len(front) < num_ret:
            indices_cnt += len(front)
            indices_select += [int(i) for i in front]
        else:
            n_keep = num_ret - indices_cnt
            F = Y_all[front]

            from pymoo.util.misc import find_duplicates
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]

            _F = F[is_unique]
            _d = calc_crowding_distance(_F)

            d = np.zeros(len(front))
            d[is_unique] = _d 
            I = np.argsort(d)[-n_keep:]
            indices_select += [int(i) for i in I]
            break
        
    return indices_select

def calc_crowding_distance(F) -> np.ndarray:

    if isinstance(F, list) or isinstance(F, np.ndarray):
        F = torch.tensor(F).to(**tkwargs)

    n_points, n_obj = F.shape

    # sort each column and get index
    I = torch.argsort(F, dim=0, descending=False)

    # sort the objective space values for the whole matrix
    F_sorted = torch.gather(F, 0, I)

    # calculate the distance from each point to the last and next
    inf_tensor = torch.full((1, n_obj), float('inf'), device=F.device, dtype=F.dtype)
    neg_inf_tensor = torch.full((1, n_obj), float('-inf'), device=F.device, dtype=F.dtype)
    dist = torch.cat([F_sorted, inf_tensor], dim=0) - torch.cat([neg_inf_tensor, F_sorted], dim=0)

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = torch.max(F_sorted, dim=0).values - torch.min(F_sorted, dim=0).values
    norm[norm == 0] = float('nan')

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist[:-1], dist[1:]
    dist_to_last, dist_to_next = dist_to_last / norm, dist_to_next / norm

    # if we divide by zero because all values in one column are equal replace by none
    dist_to_last[torch.isnan(dist_to_last)] = 0.0
    dist_to_next[torch.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = torch.argsort(I, dim=0, descending=False)
    crowding_dist = torch.sum(
        torch.gather(dist_to_last, 0, J) + torch.gather(dist_to_next, 0, J),
        dim=1
    ) / n_obj

    return crowding_dist.detach().cpu().numpy()



class ParetoSetModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj):
        super(ParetoSetModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
    
        self.fc1 = nn.Linear(self.n_obj, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_dim)
    
    def forward(self, pref):

        x = torch.relu(self.fc1(pref))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = torch.sigmoid(x) 
        
        return x

def generate_preference(y, z):
    diff = y - z
    w = 1 / diff
    w /= np.sum(w)
    return w

def psmodel_init(args, n_dim, n_obj, X_nds,Y_nds,preferences, z, nepoch=400):
        
    psmodel = ParetoSetModel(n_dim, n_obj)
    psmodel.to(args.device)
    psmodel = psmodel

    preferences_tensor = torch.tensor(preferences, dtype=torch.float32).to(args.device)
    x_tensor = torch.tensor(X_nds, dtype=torch.float32).to(args.device)


    criterion = nn.MSELoss()
    optimizer_pretrain = optim.Adam(psmodel.parameters(), lr=0.001)
    for epoch in range(nepoch):
        optimizer_pretrain.zero_grad()
        outputs = psmodel(preferences_tensor)
        loss = criterion(outputs, x_tensor)
        loss.backward()
        optimizer_pretrain.step()
        if epoch % 300 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return psmodel


def calc_coef_lcb(dhs_models, uc_li, x_tensor, n_obj, args):

    def normalize(uc, energy):
        if isinstance(energy, torch.Tensor):
            max_val = torch.tensor(uc.max).to(energy.device)
            min_val = torch.tensor(uc.min).to(energy.device)
        else:
            max_val = uc.max
            min_val = uc.min
        
        normalized_energy = (max_val - energy) / (max_val - min_val + 1e-8)
        return normalized_energy
    
    uc_e_list = []
    for obj_idx in range(n_obj):
        energy = dhs_models[obj_idx](x_tensor)
        uc_e = normalize(uc_li[obj_idx], energy)
        if args.no_energy:
            uc_e = torch.ones_like(uc_e)

        uc_e_list.append(uc_e)
    
    if isinstance(x_tensor, torch.Tensor):
        coef_lcb = torch.cat(uc_e_list, dim=1)
    else:
        coef_lcb = torch.cat([torch.tensor(uc_e) for uc_e in uc_e_list], dim=1)
    coef_lcb = torch.cat(uc_e_list, dim=1)
    return coef_lcb

    
def psmodel_train(args, psmodel, dhs_models_,model_, uc_li, n_obj, z, t_steps =100, n_steps=300, risk_ratio=2, vae=None, pca=None):
    for i in range(len(dhs_models_)):
        dhs_models_[i] = dhs_models_[i].float()
    if vae:
        def model(x):
            real_x = vae.decode(x)
            return model_(real_x)
        dhs_models = []
        for dhs_model in dhs_models_:
            def dhs_model_(x):
                real_x = vae.decode(x)
                return dhs_model(real_x)
            dhs_models.append(dhs_model_)
    elif pca:
        def model(x):
            real_x = pca.inverse_transform(x)
            return model_(real_x)
        dhs_models = []
        for dhs_model in dhs_models_:
            def dhs_model_(x):
                real_x = pca.inverse_transform(x)
                return dhs_model(real_x)
            dhs_models.append(dhs_model_)
    else:
        model = model_
        dhs_models = dhs_models_

    n_obj = len(dhs_models)

    
    optimizer = torch.optim.Adam(psmodel.parameters(), lr=1e-3)
    for t_step in tqdm(range(n_steps), desc="n_steps:"):
        psmodel.train() 
        if t_step < t_steps or args.no_bilevel:            #args.t_steps:
            alpha = np.ones(n_obj) 
            pref = np.random.dirichlet(alpha, args.n_pref_update)
            pref_vec = torch.tensor(pref).to(args.device).float() + 1e-6
            x = psmodel(pref_vec)
        else:
            pref_vec = torch.tensor(pref_vec.cpu().detach().numpy()).to(args.device)
            optimizer_ = optim.Adam([pref_vec], lr=args.lr)
            grad_list = []
            for obj_idx in range(n_obj):
                pref_vec = pref_vec.detach()
                pref_vec.requires_grad_(True)
                x = psmodel(pref_vec)
                value_ = model(x)[:, obj_idx]
                grad = autograd.grad(value_.sum(), pref_vec)[0]
                grad_list.append(grad)
            value_grad_lambda = torch.stack(grad_list, dim=1)
            risk_factor = calc_coef_lcb(dhs_models, uc_li, x.detach(), n_obj, args)
            value_grad_lambda *= risk_factor.unsqueeze(2)
            value = model(x).detach()
            pref_vec = pref_vec.detach()
            tch_idx = torch.argmax((pref_vec) * (value - z), axis = 1)
            tch_idx_mat = [torch.arange(len(tch_idx)),tch_idx]
            pref_vec = pref_vec.detach()
            tch_grad_lambda = (
                                (value-z)[tch_idx_mat].unsqueeze(1)
                                + (pref_vec)[tch_idx_mat].view(args.n_pref_update,1)*value_grad_lambda[tch_idx_mat] 
                            ) + 0.01 * torch.sum(value_grad_lambda, axis = 1)
            tch_grad_lambda = - tch_grad_lambda
            optimizer_.zero_grad()
            tch_aug = torch.sum(tch_grad_lambda.detach() * pref_vec)
            optimizer_.zero_grad()
            optimizer_.step()
        pref_vec = torch.tensor(pref_vec.cpu().detach().numpy()).to(args.device)
        
        x = psmodel(pref_vec).detach()
        grad_list = []
        for obj_idx in range(n_obj):
            x.grad = None
            x.requires_grad_(True)
            value_ = model(x)[:, obj_idx]
            grad = autograd.grad(value_.sum(), x)[0]
            grad_list.append(grad)
        
        value_grad = torch.stack(grad_list, dim=1) 
        value = model(x).detach()
        value = value.detach()
        risk_factor = calc_coef_lcb(dhs_models, uc_li, x.detach(), n_obj, args)
        tch_idx = torch.argmax((pref_vec) * (value - z), axis = 1)
        tch_idx_mat = [torch.arange(len(tch_idx)),tch_idx]
        tch_grad = (pref_vec)[tch_idx_mat].view(args.n_pref_update,1) *  value_grad[tch_idx_mat] + 0.01 * torch.sum(pref_vec.unsqueeze(2)*value_grad, axis = 1) 
        delta = torch.norm(tch_grad)
        # if t_step % 50 == 0:
        #     print('delta:',delta)
        optimizer.zero_grad()
        psmodel(pref_vec).backward(tch_grad)
        optimizer.step()



def select_random(num_select, best_subset_list, X_candidates_np, YY_candidate, do_nds=False):
    valid_index = np.array([i for i in range(len(YY_candidate)) if i not in np.array(best_subset_list).flatten()])
    if do_nds:
        nds = NonDominatedSorting()
        idx_nds = nds.do(YY_candidate)[0]
        X_candidates_np = X_candidates_np[idx_nds]
        YY_candidate = YY_candidate[idx_nds]
    if len(YY_candidate) > 0:
        num_select = min(num_select, len(valid_index))
        indexes = np.random.choice(valid_index, num_select, replace=False)
        best_subset_list.extend([[indexes[i]] for i in range(len(indexes))])
    return best_subset_list

def select_hv(max_hv_select, best_subset_list, X_candidates_np, YY_candidate, nadir_point,taskname):
    select_hv._history = []
    number_hv = 0
    full_deleted_indexes = set()
    res_indexs = set(range(len(YY_candidate))) - set(np.array(best_subset_list).flatten())
    selected_deleted_indexes = set()
    Y_p = nadir_point
    
    
    best_hv_value = -np.inf
    for b in tqdm(range(max_hv_select), desc="choose candidate-hv"):
        if number_hv >= max_hv_select or len(res_indexs) == 0:
            break  
        select_hv._history.append(Y_p.copy())
        best_subset = None
        for k in res_indexs.copy():
            if k in full_deleted_indexes:
                continue
            Y_subset = YY_candidate[[k], :]  
            Y_comb = np.vstack([Y_p, Y_subset])
            hv_value_subset = hv(nadir_point, Y_comb, taskname)
            if hv_value_subset > best_hv_value:
                best_hv_value = hv_value_subset
                best_subset = [k]
                
        if best_subset is not None:
            Y_p = np.vstack([Y_p, YY_candidate[best_subset]])
            best_subset_list.append(best_subset)
            number_hv = number_hv + 1
            full_deleted_indexes.update(best_subset)
            selected_deleted_indexes.update(best_subset)
            res_indexs.difference_update(set(np.array(best_subset, dtype=int).flatten()))
        else:
            break
    return best_subset_list

def select_igdoff(max_igd_select, best_subset_list, X_candidates_np, YY_candidate, Y_nds):
    select_igdoff._history = []
    number_igd = 0
    full_deleted_indexes = set()
    res_indexs = set(range(len(YY_candidate))) - set(np.array(best_subset_list).flatten())
    selected_deleted_indexes = set()

    n_obj = len(YY_candidate[0])
    idx_edge = []
    for i in range(n_obj):
        idx_edge.append(np.argsort(YY_candidate[:, i])[:5])
    idx_edge = np.unique(np.array(idx_edge).flatten())
    idx_edge = np.array(idx_edge)
    best_subset_list += [[i] for i in idx_edge]
    
    Y_p = YY_candidate[np.array(best_subset_list, dtype=int).flatten()]
    Y_p = np.vstack([Y_p, YY_candidate[idx_edge]])
    
    best_igd_value = np.inf
    for b in tqdm(range(max_igd_select), desc="choose candidate-igdoff"):
        if number_igd >= max_igd_select or len(res_indexs) == 0:
            break  
        select_igdoff._history.append(Y_p.copy())
        
        best_subset = None
        for k in res_indexs.copy():
            if k in full_deleted_indexes:
                continue
            Y_subset = YY_candidate[[k], :]  
            Y_comb = np.vstack([Y_p, Y_subset])
            igd_value_subset = igd_off(Y_nds, Y_comb)
            if igd_value_subset < best_igd_value:
                best_igd_value = igd_value_subset
                best_subset = [k]
                
        if best_subset is not None:
            Y_p = np.vstack([Y_p, YY_candidate[best_subset]])
            best_subset_list.append(best_subset)
            number_igd = number_igd + 1
            full_deleted_indexes.update(best_subset)
            selected_deleted_indexes.update(best_subset)
            res_indexs.difference_update(set(np.array(best_subset, dtype=int).flatten()))
        else:
            break
    return best_subset_list