import sktime
from sktime.datasets import load_from_tsfile
from sktime.transformations.panel.channel_selection import ElbowClassPairwise, ElbowClassSum
from ztime import ZTime, helper
import os
import pandas as pd
import warnings
import numpy as np
import copy
from scipy.stats import norm
from pyts.approximation import SymbolicAggregateApproximation, PiecewiseAggregateApproximation
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

def _uniform_bins(sample_min, sample_max, n_samples, n_bins):
    bin_edges = np.empty((n_bins - 1, n_samples))
    for i in range(n_samples):
        bin_edges[:, i] = np.linspace(
            sample_min[i], sample_max[i], n_bins + 1)[1:-1]
    return bin_edges

def cal_edges(X, n_bins):
    n_samples, n_timestamps = X.shape
    sample_min, sample_max = np.min(X, axis=1), np.max(X, axis=1)
    bin_edges_norm = norm.ppf(np.linspace(0, 1, n_bins + 1)[1:-1])
    bin_edges_uni = _uniform_bins(sample_min, sample_max, n_samples, n_bins).T
    bin_edges_ed = np.percentile(X, np.linspace(0, 100, n_bins + 1)[1:-1], axis=1).T
    return bin_edges_norm, bin_edges_uni, bin_edges_ed

def chi2(A, B):
    return 0.5 * np.sum(((A - B) ** 2) / (A + B))
    # return np.sum((A-B)**2)
def detect_knee_point(values, indices):
    """
    From:
    https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    """
    # get coordinates of all the points
    #print(values)
    #print(indices)
    n_points = len(values)
    #print(n_points)
    all_coords = np.vstack((range(n_points), values)).T
    # get the first point
    first_point = all_coords[0]
    # get vector between first and last point - this is the line
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    vec_from_first = all_coords - first_point
    scalar_prod = np.sum(
        vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    # distance to line is the norm of vec_to_line
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    # knee/elbow is the point with max distance value
    knee_idx = np.argmax(dist_to_line)
    knee  = values[knee_idx] 
    #print(f"Knee Value: {values[knee_idx]}, {knee_idx}") 
    
    best_dims = [idx for (elem, idx) in zip(values, indices) if elem>knee]
    # print(dist_to_line)
    # print(best_dims, knee_idx)
    if len(best_dims)==0:
        return [knee_idx], knee_idx
        print(f"Knee Value: {values[knee_idx]}, {knee_idx}") 
        
    return best_dims, knee_idx

def chooseDimensionsHistogram(data, n_bins = 5):
    X_new = []

    sax = SymbolicAggregateApproximation(n_bins = n_bins, strategy='normal')

    for dim in range(data["TRAIN"]["X"].shape[1]):
        #transform the data into the understandable form
        
        #data_transformed = np.array([i.to_numpy() for i in data[dim]])
        data_transformed = data["TRAIN"]["X"][:, dim, :]
        X_train = sax.fit_transform(data_transformed)
        X_new.append(X_train)

    X_new_final = np.moveaxis(X_new, [0], [1])

    avg_hist = []
    for dim in range(X_new_final.shape[1]):
        dim_list = []
        for i in range(X_new_final.shape[0]): 
            dim_list.append(pd.Series(X_new_final[i][dim]).value_counts())
        avg_hist.append(pd.DataFrame(dim_list).mean(axis=0))

    dims = []

    for dim in range(X_new_final.shape[1]):
        dims.append([])
        for dim2 in range(X_new_final.shape[1]):
            dims[dim].append(chi2(avg_hist[dim], avg_hist[dim2]))

    # plt.plot(np.sort(np.array(dims).mean(axis=0))[::-1])
    original_indices = np.argsort(np.array(dims).mean(axis=0))[::-1]
    indices, knee = detect_knee_point(np.sort(np.array(dims).mean(axis=0))[::-1], range(len(dims)))
    indices = original_indices[indices]
    print(indices)
    # indices, knee = detect_knee_point(np.array(dims).mean(axis=0), range(X_new_final.shape[1]))
    data["TRAIN"]["X"] = data["TRAIN"]["X"][:,indices,:]
    data["TEST"]["X"] = data["TEST"]["X"][:,indices,:]
    
    return data

MV_path = "Multivariate_ts"
all = '''RacketSports'''
DSET_NAMES = [
    # 'ArticularyWordRecognition',
    # 'AtrialFibrillation',
    # 'BasicMotions',
    ##'CharacterTrajectories',
    # 'Cricket',
    #'DuckDuckGeese', #19065
    # 'ERing',
    #'EigenWorms', #3575
    # 'Epilepsy',
    # 'EthanolConcentration',
    'FaceDetection', #45779
    # 'FingerMovements',
    # 'HandMovementDirection',
    # 'Handwriting',
    #'Heartbeat', #4485
    ##'InsectWingBeat',
    ##'JapaneseVowels',
    # 'LSST',
    # 'Libras',
    #'MotorImagery', #11920
    # 'NATOPS',
    #'PEMS-SF', #152832
    # 'PenDigits',
    #'PhonemeSpectra', #7393
    #'RacketSports',
    #'SelfRegulationSCP1',
    #'SelfRegulationSCP2',
    ##'SpokenArabicDigits',
    #'StandWalkJump',
    #'UWaveGestureLibrary',
]

results = {}

for dataset in DSET_NAMES:
    print(dataset)
    data = {}
    data["TRAIN"], data["TEST"] = {}, {}

    DATA_PATH = os.path.join(MV_path, dataset)

    data["TRAIN"]["X"], data["TRAIN"]["y"] = load_from_tsfile(
        os.path.join(DATA_PATH, dataset+"_TRAIN.ts"),  return_data_type="numpy3d"
    )
    data["TEST"]["X"], data["TEST"]["y"] = load_from_tsfile(
        os.path.join(DATA_PATH, dataset+"_TEST.ts"),  return_data_type="numpy3d"
    )
    data_2 = copy.deepcopy(data)
    data_4 = copy.deepcopy(data)

    cs = ElbowClassPairwise()
    cs.fit(data_2["TRAIN"]["X"], data_2["TRAIN"]["y"])
    data_2["TRAIN"]["X"] = cs.transform(data_2["TRAIN"]["X"])
    data_2["TEST"]["X"] = cs.transform(data_2["TEST"]["X"])

    cs = ElbowClassSum()
    cs.fit(data_4["TRAIN"]["X"], data_4["TRAIN"]["y"])
    data_4["TRAIN"]["X"] = cs.transform(data_4["TRAIN"]["X"])
    data_4["TEST"]["X"] = cs.transform(data_4["TEST"]["X"])

    if dataset not in results:
        results[dataset] = []

    for split in [2,4]:
        for reducedFeature in [True, False]:
            
            #before dimensionality reduction
            r1 =[dataset, reducedFeature, split]+list(helper.simpleTrial(data, alphabet_size=5, alphabet_size_slope=5, window_size=5, split_no=split, step=10, multivariate=True, reducedFeature=reducedFeature, varyingLength=False))
            print(r1), results[dataset].append(r1)
            #after dimensionality reduction (ECP)
            r2 = [dataset, reducedFeature, split]+list(helper.simpleTrial(data_2, alphabet_size=5, alphabet_size_slope=5, window_size=5, split_no=split, step=10, multivariate=True,  reducedFeature=reducedFeature, varyingLength=False))
            print(r2), results[dataset].append(r2)
            #after split+ECP
            r3 = [dataset, reducedFeature, split]+list(helper.simpleTrial(data, alphabet_size=5, alphabet_size_slope=5, window_size=5, split_no=split, step=10, multivariate=True,  reducedFeature=reducedFeature, varyingLength=False, dim_selection="ECP"))
            print(r3), results[dataset].append(r3)
            #after dimensionality reduction (ECS)
            r4 = [dataset, reducedFeature, split]+list(helper.simpleTrial(data_4, alphabet_size=5, alphabet_size_slope=5, window_size=5, split_no=split, step=10, multivariate=True,  reducedFeature=reducedFeature, varyingLength=False))
            print(r4), results[dataset].append(r4)
            #after split+ECS
            r5 = [dataset, reducedFeature, split]+list(helper.simpleTrial(data, alphabet_size=5, alphabet_size_slope=5, window_size=5, split_no=split, step=10, multivariate=True,  reducedFeature=reducedFeature, varyingLength=False, dim_selection="ECS"))
            print(r5), results[dataset].append(r5)
            
