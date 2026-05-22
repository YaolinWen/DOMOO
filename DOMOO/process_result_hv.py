# process_result hv

import os 
import re 
import pandas as pd 
import numpy as np 
import datetime 

# from off_moo_bench.task_set import * 


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
    # "c10mop4": "C10MOP4-Exact-v0",
    # "c10mop5": "C10MOP5-Exact-v0",
    # "c10mop6": "C10MOP6-Exact-v0",
    # "c10mop7": "C10MOP7-Exact-v0",
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
    # "in1kmop9": "IN1KMOP9-Exact-v0",
}

MONASLogitsDict = {
    "nb201_test": "NASBench201Test-Exact-v0",
}

MOCOPermutationDict = {
    "motsp_20": "BiTSP20-Exact-v0",
    "motsp_50": "BiTSP50-Exact-v0",
    "motsp_100": "BiTSP100-Exact-v0",
    "motsp_500": "BiTSP500-Exact-v0",
    "motsp3obj_20": "TriTSP20-Exact-v0",
    "motsp3obj_50": "TriTSP50-Exact-v0",
    "motsp3obj_100": "TriTSP100-Exact-v0",
    "mocvrp_20": "BiCVRP20-Exact-v0",
    "mocvrp_50": "BiCVRP50-Exact-v0",
    "mocvrp_100": "BiCVRP100-Exact-v0",
    "mokp_50": "BiKP50-Exact-v0",
    "mokp_100": "BiKP100-Exact-v0",
    "mokp_200": "BiKP200-Exact-v0",
    
    "bi_tsp_20": "BiTSP20-Exact-v0",
    "bi_tsp_50": "BiTSP50-Exact-v0",
    "bi_tsp_100": "BiTSP100-Exact-v0",
    "bi_tsp_500": "BiTSP500-Exact-v0",
    "tri_tsp_20": "TriTSP20-Exact-v0",
    "tri_tsp_50": "TriTSP50-Exact-v0",
    "tri_tsp_100": "TriTSP100-Exact-v0",
    "bi_cvrp_20": "BiCVRP20-Exact-v0",
    "bi_cvrp_50": "BiCVRP50-Exact-v0",
    "bi_cvrp_100": "BiCVRP100-Exact-v0",
    "bi_kp_50": "BiKP50-Exact-v0",
    "bi_kp_100": "BiKP100-Exact-v0",
    "bi_kp_200": "BiKP200-Exact-v0",
}

MOCOContinuousDict = {
    "portfolio": "Portfolio-Exact-v0"
}

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
    # "re41": "RE41-Exact-v0",
    # "re42": "RE42-Exact-v0",
    # "re61": "RE61-Exact-v0",
}

SyntheticFunction = list(SyntheticFunctionDict.values())
MONASSequence = list(MONASSequenceDict.values())
MONASLogits = list(MONASLogitsDict.values())
MOCOPermutation = list(MOCOPermutationDict.values())
MOCOContinuous = list(MOCOContinuousDict.values())
MORL = list(MORLDict.values())
ScientificDesignContinuous = list(ScientificDesignContinuousDict.values())
ScientificDesignSequence = list(ScientificDesignSequenceDict.values())
RESuite = list(RESuiteDict.values())

MONAS = MONASSequence + MONASLogits
MOCO = MOCOPermutation + MOCOContinuous
ScientificDesign = ScientificDesignContinuous + ScientificDesignSequence

ALLTASKS = SyntheticFunction + MONAS + MOCO + MORL + ScientificDesign + RESuite
ALLTASKSDICT = {
    **SyntheticFunctionDict,
    **MONASSequenceDict,
    **MONASLogitsDict,
    **MOCOPermutationDict,
    **MOCOContinuousDict,
    **MORLDict,
    **ScientificDesignContinuousDict,
    **ScientificDesignSequenceDict,
    **RESuiteDict,
}

REV_ALLTASKSDICT= dict()
for key in ALLTASKSDICT:
    val = ALLTASKSDICT[key]
    # val = val.lower().split('-')[0]
    REV_ALLTASKSDICT[val] = key

REV_ALLTASKSDICT

ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
ts_name = f'{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}'

BASE_PATH = os.path.abspath(".")
RESULT_DIR = os.path.join(BASE_PATH, "results")
assert os.path.exists(RESULT_DIR), "Please run your experiments first"

HV_RESULT_DIR = os.path.join(BASE_PATH, 'dst','hv',"hv_results", ts_name)
HV_LATEST_DIR = os.path.abspath(os.path.join(HV_RESULT_DIR, "..", "latest"))
os.makedirs(HV_RESULT_DIR, exist_ok=True)
os.makedirs(HV_LATEST_DIR, exist_ok=True)

AVG_RANK_RESULT_DIR = os.path.join(BASE_PATH,'dst','hv', "average_rank_results", ts_name)
AVG_RANK_LATEST_DIR = os.path.abspath(os.path.join(AVG_RANK_RESULT_DIR, "..", "latest"))
os.makedirs(AVG_RANK_RESULT_DIR, exist_ok=True)
os.makedirs(AVG_RANK_LATEST_DIR, exist_ok=True)

MODEL2MODES = {
    "End2End": ["Vallina", "GradNorm", "PcGrad"], 
    "MultiHead": ["Vallina", "GradNorm", "PcGrad"], 
    "MultipleModels": ["Vallina", "COM", "IOM", "RoMA", "ICT", "TriMentoring"], 
    "MOBO": ["Vallina", "ParEGO", "JES"],
    "DOMOO": ["Vallina"],
}

TASK_SET_PARTITION = {
    "Synthetic": SyntheticFunction,
    "MONAS": MONAS,
    "MORL": MORL,
    # "MOCO": MOCO,
    "Sci-Design": ScientificDesign,
    "RE Suite": RESuite + MOCOContinuous,
}


def find_and_read_latest_csv(root_dir, target_filename="hv_results.csv"):
    results = {}
    # iterate over roodt_dir
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            # use regular expression to match seed and timestamp
            match = re.search(r'seed(\d+).*?(\d{4}-\d{1,2}-\d{1,2}_\d{1,2}-\d{1,2}-\d{1,2})', dir_name)
            if match:
                seed = match.group(1)
                timestamp_str = match.group(2)
                timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
    
                file_path = os.path.join(root, dir_name, target_filename)
                # if file exists, decide whether to update according to timestamp
                if os.path.exists(file_path):
                    if seed not in results or results[seed][1] < timestamp:
                        results[seed] = (file_path, timestamp)

    # read every latest file
    return {key: pd.read_csv(result[0]) for key, result in results.items()}

def get_statistics(hv_array: np.ndarray):
    hv_array = hv_array.squeeze() 
    return hv_array.mean(), hv_array.std() 

def highlight_within_one_std(s, ascending: bool=True):
    if s.isna().all():
        return s
    s_copy = s.dropna().copy()
    tmp = s.dropna().apply(lambda x: float(x.split('$\pm$')[0].strip()) +
                           (1 if ascending else -1) * float(x.split('$\pm$')[1].strip()))
    
    sorted_tmp = tmp.sort_values(ascending=ascending)
    best_one = sorted_tmp.head(1).index
    
    best_mean = float(s_copy[best_one[0]].split('$\pm$')[0].strip())
    best_std = float(s_copy[best_one[0]].split('$\pm$')[1].strip())
    
    new_s = s.copy()
    for index, value in s_copy.items():
        mean = float(value.split('$\pm$')[0].strip())
        std = float(value.split('$\pm$')[1].strip())
        
        if (mean + std) >= best_mean or mean >= best_mean - best_std:
            new_s[index] = f"\\textbf{{{mean:.2f} $\pm$ {std:.2f}}}"  \
                if index != "$\mathcal{D}$(best)" else f"\\textbf{{{mean:.2f}}}"
        else:
            new_s[index] = f"{mean:.2f} $\pm$ {std:.2f}" if index != "$\mathcal{D}$(best)" else f"{mean:.2f}"
    
    return new_s

def highlight_best_two(s, ascending: bool=True):
    if s.isna().all():
        return s
    s_copy = s.dropna().copy()
    tmp = s.dropna().apply(lambda x: float(x.split('$\pm$')[0].strip()) +
                           (1 if ascending else -1) * float(x.split('$\pm$')[1].strip()))
    
    s_copy = s.dropna().copy()
    tmp = s.dropna().apply(lambda x: float(x.split('$\pm$')[0].strip()) +
                           (1 if ascending else -1) * float(x.split('$\pm$')[1].strip()))
    
    sorted_tmp = tmp.sort_values(ascending=ascending)
    best_two = sorted_tmp.head(2).index
    
    first_mean = float(s_copy[best_two[0]].split('$\pm$')[0].strip())
    first_std = float(s_copy[best_two[0]].split('$\pm$')[1].strip())
    
    second_mean = float(s_copy[best_two[1]].split('$\pm$')[0].strip())
    second_std = float(s_copy[best_two[1]].split('$\pm$')[1].strip())
    
    new_s = s.copy()
    
    for index, value in s_copy.items():
        mean = float(value.split('$\pm$')[0].strip())
        std = float(value.split('$\pm$')[1].strip())
        
        new_s[index] = f"{mean:.2f} $\pm$ {std:.2f}"
    
    if len(best_two) > 0:
        new_s[best_two[0]] = f"\\textbf{{{first_mean:.2f} $\pm$ {first_std:.2f}}}"
    if len(best_two) > 1:
        new_s[best_two[1]] = f"\\underline{{{second_mean:.2f} $\pm$ {second_std:.2f}}}"
    
    return new_s

def read_hypervolume_data(current_results_dir, percentile):
    all_csv_files = list(find_and_read_latest_csv(current_results_dir, "hv_results.csv").values())
    if not all_csv_files:
        return None, None
    
    hv_data = np.array([csv_file[f"hypervolume/{percentile}"][0] for csv_file in all_csv_files])
    return get_statistics(hv_data)

def create_hv_dataframe(task_set, percentiles):
    algo_entries = ["$\mathcal{D}$(best)"] + [f"{model} + {mode}" for model, modes in MODEL2MODES.items() for mode in modes]
    task_set_short = [task.split('-')[0] for task in task_set]
    hv_dfs = {p: pd.DataFrame(index=algo_entries, columns=task_set_short) for p in percentiles}
    for df in hv_dfs.values():
        df.index.name = 'Methods'
    return hv_dfs

def fill_hv_dataframe(task_set, hv_dfs, percentiles):
    d_best_values = {}
    for task in task_set:
        task_entry = task.split('-')[0]
        for model, modes in MODEL2MODES.items():
            for mode in modes:
                folder_name = f"{model}-{mode}-{task}"
                current_results_dir = os.path.join(RESULT_DIR, folder_name)
                if not os.path.exists(current_results_dir):
                    continue

                for percentile in percentiles:
                    mean, std = read_hypervolume_data(current_results_dir, percentile)
                    if mean is None:
                        continue
                    algo_entry = f"{model} + {mode}"
                    hv_dfs[percentile][task_entry][algo_entry] = f"{mean} $\pm$ {std}"
                    
                all_csv_files = list(find_and_read_latest_csv(current_results_dir, "hv_results.csv").values())
                if all_csv_files:
                    d_best_values[task_entry] = f"{all_csv_files[0]['hypervolume/D(best)'].item()} $\pm$ 0.0"
                    
    for percentile in percentiles:
        for task_entry, value in d_best_values.items():
            hv_dfs[percentile][task_entry]["$\mathcal{D}$(best)"] = value
                    
def calculate_avg_rank_for_single_df(s: pd.DataFrame):
    s_copy = s.copy()
    mean_df = s_copy.applymap(lambda x: float(x) if pd.notna(x) else x)
    ranks = mean_df.rank(axis=0, method='average', na_option='keep', ascending=False)
    mean_ranks = ranks.mean(axis=1, skipna=True)
    return mean_ranks

def calculate_mean_std(seed2rank_df: dict):
    all_seeds = list(seed2rank_df.keys())
    all_rank_df = list(seed2rank_df.values())
    
    for i in range(1, len(all_rank_df)):
        if not (all_rank_df[i].index.equals(all_rank_df[i-1].index) and all_rank_df[i].columns.equals(all_rank_df[i-1].columns)):
            raise ValueError(f"Indices or columns do not match between DataFrame of seed {all_seeds[i]} and DataFrame {all_seeds[i-1]}.")
    
    result_df = pd.DataFrame(index=all_rank_df[0].index, columns=all_rank_df[0].columns)
    for col in all_rank_df[0].columns:
        if np.issubdtype(all_rank_df[0][col].dtype, np.number):
            for i in all_rank_df[0].index:
                vals = np.array([rank_df.at[i, col] for rank_df in all_rank_df])
                is_valid = np.where(~np.isnan(vals))[0]
                
                if len(is_valid) == 0:
                    result_df.at[i, col] = np.nan 
                elif len(is_valid) == 1:
                    result_df.at[i, col] = f"{vals[is_valid].item()} $\pm$ 0.00"
                else:
                    mean = np.mean(vals[is_valid])
                    std = np.std(vals[is_valid])
                    result_df.at[i, col] = f"{mean} $\pm$ {std}"
                    
    return result_df


def calculate_performance():
    percentiles = ['100th', '75th', '50th']
    
    for task_type, task_set in TASK_SET_PARTITION.items():
        hv_dfs = create_hv_dataframe(task_set, percentiles)
        fill_hv_dataframe(task_set, hv_dfs, percentiles)

        for percentile in percentiles:
            hv_df = hv_dfs[percentile].apply(highlight_within_one_std, ascending=False)
            hv_df.to_csv(os.path.join(HV_RESULT_DIR, f"{task_type}-HV-{percentile}.csv"))
            hv_df.to_csv(os.path.join(HV_LATEST_DIR, f"{task_type}-HV-{percentile}.csv"))
            


def calculate_mean_rank():
    algo_entries = ["$\mathcal{D}$(best)"] + [f"{model} + {mode}" for model, modes in MODEL2MODES.items() for mode in modes]
    
    seed2rank_100th = {}
    seed2rank_75th = {}
    seed2rank_50th = {}
    
    seed2allhv_100th = {}
    seed2allhv_75th = {}
    seed2allhv_50th = {}
    
    for task_type, task_set in TASK_SET_PARTITION.items():
        task_set_short = [task.split('-')[0] for task in task_set]
        
        seed2hv_100th = {}
        seed2hv_75th = {}
        seed2hv_50th = {}
        
        for task in task_set:
            task_entry = task.split('-')[0]
            for model, modes in MODEL2MODES.items():
                for mode in modes:
                    folder_name = f"{model}-{mode}-{task}"
                    algo_entry = f"{model} + {mode}"
                    current_results_dir = os.path.join(RESULT_DIR, folder_name)
                    if not os.path.exists(current_results_dir):
                        continue
                    
                    seed2csv_files = find_and_read_latest_csv(current_results_dir, "hv_results.csv")
                    if len(seed2csv_files) == 0:
                        continue
                    
                    for seed, csv_file in seed2csv_files.items():
                        if seed not in seed2hv_100th.keys():
                            hv_df_100th = pd.DataFrame(index=algo_entries, columns=task_set_short)
                            hv_df_100th.index.name = 'Methods'
                            seed2hv_100th[seed] = hv_df_100th
                        if seed not in seed2hv_75th.keys():
                            hv_df_75th = pd.DataFrame(index=algo_entries, columns=task_set_short)
                            hv_df_75th.index.name = 'Methods'
                            seed2hv_75th[seed] = hv_df_75th
                        if seed not in seed2hv_50th.keys():
                            hv_df_50th = pd.DataFrame(index=algo_entries, columns=task_set_short)
                            hv_df_50th.index.name = 'Methods'
                            seed2hv_50th[seed] = hv_df_50th
                        seed2hv_100th[seed][task_entry][algo_entry] = csv_file["hypervolume/100th"][0]
                        seed2hv_75th[seed][task_entry][algo_entry] = csv_file["hypervolume/75th"][0]
                        seed2hv_50th[seed][task_entry][algo_entry] = csv_file["hypervolume/50th"][0]
                        
                        seed2hv_100th[seed][task_entry]["$\mathcal{D}$(best)"] = csv_file["hypervolume/D(best)"][0]
                        seed2hv_75th[seed][task_entry]["$\mathcal{D}$(best)"] = csv_file["hypervolume/D(best)"][0]
                        seed2hv_50th[seed][task_entry]["$\mathcal{D}$(best)"] = csv_file["hypervolume/D(best)"][0]
        
        for seed, df in seed2hv_100th.items():
            if seed not in seed2rank_100th.keys():
                seed2rank_100th[seed] = pd.DataFrame(index=algo_entries, columns=list(TASK_SET_PARTITION.keys()) + ["Avg. Rank"])
                seed2rank_100th[seed].index.name = "Methods"
            rank_df = calculate_avg_rank_for_single_df(df)
            seed2rank_100th[seed][task_type] = rank_df
            
            if seed not in seed2allhv_100th.keys():
                seed2allhv_100th[seed] = pd.DataFrame(index=algo_entries)
                seed2allhv_100th[seed].index.name = "Methods"
            seed2allhv_100th[seed] = pd.concat([seed2allhv_100th[seed], df], axis=1)
        
        for seed, df in seed2hv_75th.items():
            if seed not in seed2rank_75th.keys():
                seed2rank_75th[seed] = pd.DataFrame(index=algo_entries, columns=list(TASK_SET_PARTITION.keys()) + ["Avg. Rank"])
                seed2rank_75th[seed].index.name = "Methods"
            rank_df = calculate_avg_rank_for_single_df(df)
            seed2rank_75th[seed][task_type] = rank_df
            
            if seed not in seed2allhv_75th.keys():
                seed2allhv_75th[seed] = pd.DataFrame(index=algo_entries)
                seed2allhv_75th[seed].index.name = "Methods"
            seed2allhv_75th[seed] = pd.concat([seed2allhv_75th[seed], df], axis=1)
            
        for seed, df in seed2hv_50th.items():
            if seed not in seed2rank_50th.keys():
                seed2rank_50th[seed] = pd.DataFrame(index=algo_entries, columns=list(TASK_SET_PARTITION.keys()) + ["Avg. Rank"])
                seed2rank_50th[seed].index.name = "Methods"
            rank_df = calculate_avg_rank_for_single_df(df)
            seed2rank_50th[seed][task_type] = rank_df
            
            if seed not in seed2allhv_50th.keys():
                seed2allhv_50th[seed] = pd.DataFrame(index=algo_entries)
                seed2allhv_50th[seed].index.name = "Methods"
            seed2allhv_50th[seed] = pd.concat([seed2allhv_50th[seed], df], axis=1)

    for seed, df in seed2allhv_100th.items():
        all_avg_rank_100th = calculate_avg_rank_for_single_df(df)
        seed2rank_100th[seed]["Avg. Rank"] = all_avg_rank_100th
    
    for seed, df in seed2allhv_75th.items():
        all_avg_rank_75th = calculate_avg_rank_for_single_df(df)
        seed2rank_75th[seed]["Avg. Rank"] = all_avg_rank_75th
        
    for seed, df in seed2allhv_50th.items():
        all_avg_rank_50th = calculate_avg_rank_for_single_df(df)
        seed2rank_50th[seed]["Avg. Rank"] = all_avg_rank_50th
    
    avg_rank_100th = calculate_mean_std(seed2rank_100th)
    avg_rank_100th = avg_rank_100th.apply(highlight_best_two, ascending=True)
    avg_rank_100th.to_csv(os.path.join(AVG_RANK_RESULT_DIR, "average_rank_100th.csv"))
    avg_rank_100th.to_csv(os.path.join(AVG_RANK_LATEST_DIR, "average_rank_100th.csv"))
    
    avg_rank_75th = calculate_mean_std(seed2rank_75th)
    avg_rank_75th = avg_rank_75th.apply(highlight_best_two, ascending=True)
    avg_rank_75th.to_csv(os.path.join(AVG_RANK_RESULT_DIR, "average_rank_75th.csv"))
    avg_rank_75th.to_csv(os.path.join(AVG_RANK_LATEST_DIR, "average_rank_75th.csv"))
    
    avg_rank_50th = calculate_mean_std(seed2rank_50th)
    avg_rank_50th = avg_rank_50th.apply(highlight_best_two, ascending=True)
    avg_rank_50th.to_csv(os.path.join(AVG_RANK_RESULT_DIR, "average_rank_50th.csv"))
    avg_rank_50th.to_csv(os.path.join(AVG_RANK_LATEST_DIR, "average_rank_50th.csv"))

# def process_hv():
calculate_performance()
calculate_mean_rank() 