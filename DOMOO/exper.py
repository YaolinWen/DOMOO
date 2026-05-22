import argparse, os, sys
from utils import ALLTASKSDICT, MONASSequenceDict

def set_default(args, attr, value):
    for argv in sys.argv:
        if f'--{attr}' in argv:
            return
    setattr(args, attr, value)


def get_parser():
    parser = argparse.ArgumentParser(description='Run the model with different hidden sizes')
    parser.add_argument('--n_steps', type=int, default=300)
    parser.add_argument('--psmodel_pretrain_nepoch', type=int, default=200)
    parser.add_argument('--n_pref_update', type=int, default=256)
    parser.add_argument('--results_dir', type=str, default="./result")
    parser.add_argument("--energy_save_dir", type=str, default="./energy_model")
    parser.add_argument("--multiple_model_results", type=str, default="../baseline/offline-moo/results")
    parser.add_argument("--seed",           type=int,  default=1000)
    parser.add_argument("--skip_existing",  type=str, default=False)
    
    parser.add_argument("--task", type=str, default="zdt3")
    parser.add_argument("--normalization", type=str, default=True)
    # number of iterations, and batch size per iteration
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    # parser.add_argument("--hidden_size", type=int, default=256)

    # parser.add_argument("--train_mode", type=str, default="sde")
    
    parser.add_argument("--t_steps", type=int, default=200)
    
    parser.add_argument("--train_mode", type=str, default="Vallina")
    parser.add_argument("--model", type=str, default="DOMOO")
    parser.add_argument("--model_save_dir", type=str, default='./model')
    parser.add_argument("--num_solutions", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--retrain_model", type=str, default=False)
    parser.add_argument("--forward_lr", type=float, default=1e-3)
    parser.add_argument("--forward_lr_decay", type=float, default=0.98)
    parser.add_argument("--risk_ratio", type=float, default=1e-3)
    
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--use_wandb", type=str, default=False)
    parser.add_argument("--data_preserved_ratio", type=float, default=0.2)
    parser.add_argument("--data_pruning", type=str, default=True)
    parser.add_argument("--n_candidate", type=int, default=1000)
    parser.add_argument("--n_local", type=int, default=1)
    parser.add_argument("--coef_lcb", type=float, default=0.1)
    parser.add_argument("--n_sample", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument('--init_m', type=float, default=0.05)
    parser.add_argument('--Ld_K_max', type=int, default=42)
    parser.add_argument('--Ld_K', type=int, default=64)
    # parser.add_argument('--n_steps', type=int, default=1000)
    # parser.add_argument('--n_pref_update', type=int, default=256)
    parser.add_argument('--ratios', type=float, default=0.3)
    parser.add_argument('--indicator', type=str, default="hv")
    parser.add_argument('--train_epoch', type=int, default=50)
    parser.add_argument('--no_energy', type=str, default=False)
    parser.add_argument('--no_bilevel', type=str, default=False)
    parser.add_argument('--no_igdoff', type=str, default=False)
    parser.add_argument('--no_psl', type=str, default=False)
    parser.add_argument('--no_surrogate', type=str, default=False)

    args = parser.parse_args()

    if type(args.use_wandb) == str:
        args.use_wandb = (args.use_wandb.lower() == 'true')
    if type(args.retrain_model) == str:
        args.retrain_model = (args.retrain_model.lower() == 'true')
    if type(args.data_pruning) == str:
        args.data_pruning = (args.data_pruning.lower() == 'true')
    if type(args.normalization) == str:
        args.normalization = (args.normalization.lower() == 'true')
    if type(args.skip_existing) == str:
        args.skip_existing = (args.skip_existing.lower() == 'true')
    if type(args.no_energy) == str:
        args.no_energy = (args.no_energy.lower() == 'true')
    if type(args.no_bilevel) == str:
        args.no_bilevel = (args.no_bilevel.lower() == 'true')
    if type(args.no_igdoff) == str:
        args.no_igdoff = (args.no_igdoff.lower() == 'true')
    if type(args.no_psl) == str:
        args.no_psl = (args.no_psl.lower() == 'true')
    if type(args.no_surrogate) == str:
        args.no_surrogate = (args.no_surrogate.lower() == 'true')

    if args.task in ['c10mop1', 'c10mop2', 'c10mop3', 'c10mop4', 'c10mop5', 'c10mop6', 'c10mop7', 'c10mop8', 'c10mop9', 
                    'in1kmop1', 'in1kmop2', 'in1kmop3', 'in1kmop4', 'in1kmop5', 'in1kmop6', 'in1kmop7', 'in1kmop8', 'in1kmop9',
                    'regex', 'rfp', 'zinc']:
        set_default(args, 'normalization', False)
    else:
        set_default(args, 'normalization', True)
    
    if args.task in ALLTASKSDICT.keys():
        task_name = ALLTASKSDICT[args.task]
    else:
        task_name = args.task

    
    if task_name in [ALLTASKSDICT['mo_hopper_v2']]:
        set_default(args, 'psmodel_pretrain_nepoch', 100)
        set_default(args, 't_steps', 5)
        set_default(args, 'n_steps', 10)
        set_default(args, 'Ld_K_max', 8)

    elif task_name in [ALLTASKSDICT['mo_swimmer_v2']]:
        set_default(args, 'psmodel_pretrain_nepoch', 500)
        set_default(args, 't_steps', 1)
        set_default(args, 'n_steps', 0)
        set_default(args, 'Ld_K_max', 8)
    elif task_name in MONASSequenceDict.values():
        set_default(args, 'psmodel_pretrain_nepoch', 100)
        set_default(args, 't_steps', 400)
        set_default(args, 'n_steps', 800)
        set_default(args, 'Ld_K_max', 64)
    return args