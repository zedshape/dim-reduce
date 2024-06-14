"""

This helper module has functions that help create interval structure from raw time series.

"""
from pyts.approximation import SymbolicAggregateApproximation, PiecewiseAggregateApproximation
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import RidgeClassifierCV
import time
from scipy.stats import norm
import pandas as pd

from sktime.transformations.panel.channel_selection import ElbowClassPairwise, ElbowClassSum
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from ztime import ZTime

def znorm(samples):
  return (samples - samples.mean(axis=-1)[:,...,np.newaxis]) / samples.std(axis=-1)[:,...,np.newaxis]

def createDatabase(data, window_size = 10, window_size_slope = 5, alphabet_size = 3, alphabet_size_slope = 5, glob=False, uniform=True, quantile=True, normal=True):
    alphabet_origin = 97
    
    # Hard coded 
    alphabets_1 = [chr(alphabet_origin + i) for i in range(alphabet_size)]
    alphabets_2 = [chr(alphabet_origin + i + alphabet_size) for i in range(alphabet_size)]
    alphabets_3 = [chr(alphabet_origin + i + alphabet_size*2) for i in range(alphabet_size)]
    alphabets_4 = [chr(alphabet_origin + i + alphabet_size*3) for i in range(alphabet_size)]
    alphabets_5 = [chr(alphabet_origin + i + alphabet_size*4) for i in range(alphabet_size)]
    alphabets_6 = [chr(alphabet_origin + i + alphabet_size*5) for i in range(alphabet_size)]

    # FOR SLOPES
    alphabets_7 = [chr(alphabet_origin + i + alphabet_size*6) for i in range(alphabet_size_slope)]
    alphabets_8 = [chr(alphabet_origin + i + alphabet_size*6 + alphabet_size_slope*1) for i in range(alphabet_size_slope)]
    alphabets_9 = [chr(alphabet_origin + i + alphabet_size*6 + alphabet_size_slope*2) for i in range(alphabet_size_slope)]
    alphabets_10 = [chr(alphabet_origin + i + alphabet_size*6 + alphabet_size_slope*3) for i in range(alphabet_size_slope)]

    PAA_transformer = PiecewiseAggregateApproximation(window_size=window_size)

    # Step 1. Create PAA-SAX intervals
    #
    # Raw dataset will be PAA-ed to catch the pattern robust to outliers and noises
    #

    SAX_uniform = SAXify(data,  alphabet = alphabets_1, n_bins = alphabet_size, glob=glob, strategy='uniform')
    SAX_quantile = SAXify(data,  alphabet = alphabets_2, n_bins = alphabet_size, glob=glob, strategy='quantile')
    SAX_normal = SAXify(data,  alphabet = alphabets_3, n_bins = alphabet_size, glob=glob, strategy='normal')
    
    PAA_SAX_uniform = SAXify(PAA_transformer.transform(data), alphabet = alphabets_4, n_bins = alphabet_size, glob=glob, strategy='uniform')
    PAA_SAX_quantile = SAXify(PAA_transformer.transform(data), alphabet = alphabets_5, n_bins = alphabet_size, glob=glob, strategy='quantile')
    PAA_SAX_normal = SAXify(PAA_transformer.transform(data), alphabet = alphabets_6, n_bins = alphabet_size, glob=glob, strategy='normal')

    # Step 2. Create only-SAX intervals
    #
    # derivatives will not be PAA-ed since they only have limited numbers
    # For the same reason diff does not have normal layer since it is too strong assumption
    #

    # diff the dataset
    diff_data = np.diff(data, prepend=0)
    diff_double_data = np.diff(diff_data, prepend=0)
    
    # diff does not have normal layer since it is too strong assumption
    diff_SAX_uniform = SAXify(diff_data,  alphabet = alphabets_7, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
    diff_SAX_quantile = SAXify(diff_data,  alphabet = alphabets_8, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')
    
    # double diff does not have normal layer since it is too strong assumption
    diff_double_uniform_SAX = SAXify(diff_double_data, alphabet = alphabets_9, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
    diff_double_quantile_SAX = SAXify(diff_double_data, alphabet = alphabets_10, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')

    new_database = []
    
    for row in range(data.shape[0]):
        newRow = []

        newRow += createIntervals(diff_SAX_quantile[row], end_size = diff_data.shape[-1], window_size = window_size_slope)
        newRow += createIntervals(diff_double_quantile_SAX[row], end_size = diff_double_data.shape[-1], window_size = window_size_slope)
        newRow += createIntervals(diff_SAX_uniform[row], end_size = diff_data.shape[-1], window_size = window_size_slope)
        newRow += createIntervals(diff_double_uniform_SAX[row], end_size = diff_double_data.shape[-1], window_size = window_size_slope)

        
        if normal == True:
            newRow += createIntervals(SAX_normal[row],  end_size = data.shape[-1], window_size = 1)
            newRow += createIntervals(PAA_SAX_normal[row], end_size = data.shape[-1], window_size = window_size)
        if uniform == True:
            newRow += createIntervals(SAX_uniform[row],  end_size = data.shape[-1], window_size = 1)
            newRow += createIntervals(PAA_SAX_uniform[row], end_size = data.shape[-1], window_size = window_size)
        if quantile == True:
            newRow += createIntervals(SAX_quantile[row],  end_size = data.shape[-1], window_size = 1)
            newRow += createIntervals(PAA_SAX_quantile[row], end_size = data.shape[-1], window_size = window_size)
        

        new_database.append(removeSpecificEventLabel(newRow, []))
    return new_database


def loadTopic(dir, topic):
    values = []
    labels = []
    for tt in ["TRAIN", "TEST"]:
        # Load the file
        dataset = np.loadtxt(f"{dir}/{topic}/{topic}_{tt}.txt")
        # split the labels from the dataset
        labels.append(dataset[:, 0])
        values.append(dataset[:, 1:])
    return {"TRAIN": {"X": values[0], "y": labels[0]}, "TEST": {"X": values[1], "y": labels[1]}}


    
def createIntervalsMissingInterval(data, mask, end_size, window_size=1):
    prevCol = None
    count = 0
    prevFinalCount = 0
    newRow = []
    
    for colIdx in range(len(data)):
        currentCol = data[colIdx]
        # prevColMask = False
        currColMask = mask[colIdx]
        # if currColMask == True:
        #     prevCol = None
        #     prevFinalCount = prevFinalCount + 1
        #     count = 1

        # if prevColMask == True:
        #     prevCol = currentCol
        #     prevFinalCount = prevFinalCount + 1
        #     prevColMask = mask[colIdx]
        #     count = 1
        #     continue
        # Initial condition
        if prevCol == None:
            prevCol = currentCol
            count += 1
        else:
            if currentCol == prevCol:
                count += 1
            else:
                # need to make an interval
                #print((prevCol, (prevFinalCount), prevFinalCount + (count*window_size)))
                # print(prevColMask, ((prevFinalCount), prevFinalCount + (count*window_size), prevCol))
                if prevColMask == False:
                    newRow.append(((prevFinalCount), prevFinalCount + (count*window_size), prevCol))
                #BUG: Why +1 is here?
                # prevFinalCount = (prevFinalCount + count*window_size + 1)
                prevFinalCount = (prevFinalCount + count*window_size)
                count = 1
                prevCol = currentCol
        prevColMask = mask[colIdx]

    # final condition
    if prevColMask == False:
        newRow.append((prevFinalCount, end_size, prevCol))
    
    return newRow

def createIntervals(data, end_size, window_size=1):
    prevCol = None
    count = 0
    prevFinalCount = 0
    newRow = []
    
    for currentCol in data:
        # Initial condition
        if prevCol == None:
            prevCol = currentCol
            count += 1
        else:
            if currentCol == prevCol:
                count += 1
            else:
                # need to make an interval
                #print((prevCol, (prevFinalCount), prevFinalCount + (count*window_size)))
                newRow.append(((prevFinalCount), prevFinalCount + (count*window_size), prevCol))
                #BUG: Why +1 is here?
                # prevFinalCount = (prevFinalCount + count*window_size + 1)
                prevFinalCount = (prevFinalCount + count*window_size)
                count = 1
                prevCol = currentCol

    # final condition
    newRow.append((prevFinalCount, end_size, prevCol))
    #print((prevCol, prevFinalCount, end_size))
    
    return newRow

def createMultivariateDatabaseVL(data, window_size = 10, window_size_slope = 5, alphabet_size = 3, alphabet_size_slope = 5, glob=False, uniform=True, quantile=True, normal=True, varyingLength=False, reducedFeature=False):

    # get dimension
    # row, dimension, length
    dims = len(data[0])
    diff_data = np.zeros(1)
    diff_double_data = np.zeros(1)

    SAX_PAA_info = {}
    values = {}
    # for each dimension we need to save SAX_PAA information

    # ROW, DIM, LENGTH
    step_size = alphabet_size*6 + alphabet_size_slope*4
    for dim in range(dims):
        #transform the data into the understandable form
        
        #data_transformed = np.array([i.to_numpy() for i in data[dim]])
        # without dim -> row, length 2 dimensional data
        # ROW, LENGTH
        data_transformed = [i[dim] for i in data]

        alphabets_1 = [step_size*dim + i for i in range(alphabet_size)] #0,1,2
        alphabets_2 = [step_size*dim + i + alphabet_size for i in range(alphabet_size)] #3,4,5
        alphabets_3 = [step_size*dim + i + alphabet_size*2 for i in range(alphabet_size)] #6,7,8
        alphabets_4 = [step_size*dim + i + alphabet_size*3 for i in range(alphabet_size)] #9,10,11
        alphabets_5 = [step_size*dim + i + alphabet_size*4 for i in range(alphabet_size)] #12,13,14
        alphabets_6 = [step_size*dim + i + alphabet_size*5 for i in range(alphabet_size)] #15,16,17

        # FOR SLOPES
        alphabets_7 = [step_size*dim  + i + alphabet_size*6 for i in range(alphabet_size_slope)] #18, 19
        alphabets_8 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*1 for i in range(alphabet_size_slope)] #20,21
        alphabets_9 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*2 for i in range(alphabet_size_slope)] #22,23
        alphabets_10 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*3 for i in range(alphabet_size_slope)] #24,25

        PAA_transformer = PiecewiseAggregateApproximation(window_size=window_size)

        # diff the dataset
        #diff_data = np.diff(data_transformed, prepend=0)
        diff_data = [np.diff(i, prepend=i[0]) for i in data_transformed]
        diff_double_data = [np.diff(i, prepend=i[0]) for i in diff_data]
        #diff_double_data = np.diff(diff_data, prepend=0)
        
        # Step 1. Create PAA-SAX intervals
        #
        # Raw dataset will be PAA-ed to catch the pattern robust to outliers and noises
        #

        SAX_uniform = [SAXify([a], alphabet = alphabets_1, n_bins = alphabet_size, glob=glob, strategy='uniform')[0] for a in data_transformed]
        SAX_quantile = [SAXify([a], alphabet = alphabets_2, n_bins = alphabet_size, glob=glob, strategy='quantile')[0] for a in data_transformed]
        SAX_normal = [SAXify([a], alphabet = alphabets_3, n_bins = alphabet_size, glob=glob, strategy='normal')[0] for a in data_transformed]
        
        #SAX_uniform = SAXify(data_transformed,  alphabet = alphabets_1, n_bins = alphabet_size, glob=glob, strategy='uniform')
        #SAX_quantile = SAXify(data_transformed,  alphabet = alphabets_2, n_bins = alphabet_size, glob=glob, strategy='quantile')
        #SAX_normal = SAXify(data_transformed,  alphabet = alphabets_3, n_bins = alphabet_size, glob=glob, strategy='normal')
        PAA_SAX_uniform = [SAXify(PAA_transformer.transform([a]), alphabet = alphabets_4, n_bins = alphabet_size, glob=glob, strategy='uniform')[0] for a in data_transformed]
        PAA_SAX_quantile = [SAXify(PAA_transformer.transform([a]), alphabet = alphabets_5, n_bins = alphabet_size, glob=glob, strategy='quantile')[0] for a in data_transformed]
        PAA_SAX_normal = [SAXify(PAA_transformer.transform([a]), alphabet = alphabets_6, n_bins = alphabet_size, glob=glob, strategy='normal')[0] for a in data_transformed]
        
        # PAA_SAX_uniform = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_4, n_bins = alphabet_size, glob=glob, strategy='uniform')
        # PAA_SAX_quantile = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_5, n_bins = alphabet_size, glob=glob, strategy='quantile')
        # PAA_SAX_normal = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_6, n_bins = alphabet_size, glob=glob, strategy='normal')

        # Step 2. Create only-SAX intervals
        #
        # derivatives will not be PAA-ed since they only have limited numbers
        # For the same reason diff does not have normal layer since it is too strong assumption
        
        # diff does not have normal layer since it is too strong assumption
        diff_SAX_uniform = [SAXify([a], alphabet = alphabets_7, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')[0] for a in diff_data]
        diff_SAX_quantile = [SAXify([a], alphabet = alphabets_8, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')[0] for a in diff_data]
        #diff_SAX_uniform = SAXify(diff_data, alphabet = alphabets_7, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
        #diff_SAX_quantile = SAXify(diff_data, alphabet = alphabets_8, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')
        
        # double diff does not have normal layer since it is too strong assumption
        diff_double_uniform_SAX = [SAXify([a], alphabet = alphabets_9, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')[0] for a in diff_double_data]
        diff_double_quantile_SAX = [SAXify([a], alphabet = alphabets_10, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')[0] for a in diff_double_data]
        #diff_double_uniform_SAX = SAXify(diff_double_data, alphabet = alphabets_9, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
        #diff_double_quantile_SAX = SAXify(diff_double_data, alphabet = alphabets_10, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')

        SAX_PAA_info[dim] = {"SAX_uniform": SAX_uniform, 
                             "SAX_quantile": SAX_quantile, 
                             "SAX_normal": SAX_normal, 
                             "PAA_SAX_uniform": PAA_SAX_uniform, 
                             "PAA_SAX_quantile": PAA_SAX_quantile, 
                             "PAA_SAX_normal": PAA_SAX_normal, 
                             "diff_SAX_uniform": diff_SAX_uniform, 
                             "diff_SAX_quantile": diff_SAX_quantile, 
                             "diff_double_uniform_SAX": diff_double_uniform_SAX, 
                             "diff_double_quantile_SAX": diff_double_quantile_SAX}
        
        values_original = [SAX_uniform, SAX_quantile, SAX_normal, PAA_SAX_uniform, PAA_SAX_quantile, PAA_SAX_normal]
        

        ############ technique choosing ###############
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
        
        if reducedFeature == True:
            bin_edges_norm, bin_edges_uni, bin_edges_ed = cal_edges(data_transformed, alphabet_size)

            # print(np.sqrt(np.mean((bin_edges_norm - bin_edges_uni)**2)), np.sqrt(np.mean((bin_edges_norm - bin_edges_ed)**2)))

            X_PAA = PiecewiseAggregateApproximation(window_size=window_size).transform(data_transformed)
            bin_edges_norm_PAA, bin_edges_uni_PAA, bin_edges_ed_PAA = cal_edges(X_PAA, alphabet_size)

            dists = [bin_edges_uni, bin_edges_ed, bin_edges_norm, bin_edges_uni_PAA, bin_edges_ed_PAA, bin_edges_norm_PAA]

            dims = []
            for i, dim in enumerate(dists):
                dims.append([])
                for dim2 in dists:
                    dims[i].append(np.sqrt(np.mean((dim-dim2)**2)))
            
            # print(dims)
            #plt.plot(np.sort(np.array(dims).mean(axis=0))[::-1])
            original_indices = np.argsort(np.array(dims).mean(axis=0))[::-1]
            indices, knee = detect_knee_point(np.sort(np.array(dims).mean(axis=0))[::-1], range(len(dims)))
            indices = original_indices[indices]

            #for values_original we perform bin analysis and choose the ones meeting the requirement
            values[dim] = [values_original[i] for i in indices] # choose the dimensions!
        else:
            values[dim] = values_original

    new_database = []
    
    for row in range(len(data)):
        newRow = []

        for dim in range(dims): 
            for value in values:
        # Each row, each dimension 
                # We need to add those different intervals in the same database
                newRow += createIntervals(value[row], end_size = len(data[row][0]), window_size = window_size_slope)
                # newRow += createIntervals(SAX_PAA_info[dim]["diff_double_quantile_SAX"][row], end_size = len(data[row][0]), window_size = window_size_slope)
                # newRow += createIntervals(SAX_PAA_info[dim]["diff_SAX_uniform"][row], end_size =  len(data[row][0]), window_size = window_size_slope)
                # newRow += createIntervals(SAX_PAA_info[dim]["diff_double_uniform_SAX"][row], end_size = len(data[row][0]), window_size = window_size_slope)

                
            if normal == True:
                newRow += createIntervals(SAX_PAA_info[dim]["SAX_normal"][row],  end_size = len(data[row][0]), window_size = 1)
                newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_normal"][row], end_size = len(data[row][0]), window_size = window_size)
            if uniform == True:
                newRow += createIntervals(SAX_PAA_info[dim]["SAX_uniform"][row],  end_size =  len(data[row][0]), window_size = 1)
                newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_uniform"][row], end_size = len(data[row][0]), window_size = window_size)
            if quantile == True:
                newRow += createIntervals(SAX_PAA_info[dim]["SAX_quantile"][row],  end_size = len(data[row][0]), window_size = 1)
                newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_quantile"][row], end_size = len(data[row][0]), window_size = window_size)
                
        new_database.append(sorted(newRow))
    return new_database


def createMultivariateDatabaseMissingData(data, mask, window_size = 10, window_size_slope = 5, alphabet_size = 3, alphabet_size_slope = 5, glob=False, uniform=True, quantile=True, normal=True, varyingLength=False, reducedFeature=False):
    alphabet_origin = 97
    
    # get dimension
    # row, dimension, length
    dims = data.shape[1]
    diff_data = np.zeros(1)
    diff_double_data = np.zeros(1)
    data_transformed = np.zeros(1)

    SAX_PAA_info = {}
    values = {}
    
    step_size = alphabet_size*6 + alphabet_size_slope*4
    # for each dimension we need to save SAX_PAA information
    for dim in range(dims):
        #transform the data into the understandable form
        
        #data_transformed = np.array([i.to_numpy() for i in data[dim]])
        data_transformed = data[:, dim, :]

        alphabets_1 = [step_size*dim + i for i in range(alphabet_size)] #0,1,2
        alphabets_2 = [step_size*dim + i + alphabet_size for i in range(alphabet_size)] #3,4,5
        alphabets_3 = [step_size*dim + i + alphabet_size*2 for i in range(alphabet_size)] #6,7,8
        alphabets_4 = [step_size*dim + i + alphabet_size*3 for i in range(alphabet_size)] #9,10,11
        alphabets_5 = [step_size*dim + i + alphabet_size*4 for i in range(alphabet_size)] #12,13,14
        alphabets_6 = [step_size*dim + i + alphabet_size*5 for i in range(alphabet_size)] #15,16,17

        # FOR SLOPES
        alphabets_7 = [step_size*dim  + i + alphabet_size*6 for i in range(alphabet_size_slope)] #18, 19,20
        alphabets_8 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*1 for i in range(alphabet_size_slope)] #21,22,23
        alphabets_9 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*2 for i in range(alphabet_size_slope)] #24,25,26
        alphabets_10 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*3 for i in range(alphabet_size_slope)] #27,28,29

        PAA_transformer = PiecewiseAggregateApproximation(window_size=window_size)

        # diff the dataset
        diff_data = np.diff(data_transformed, prepend=0)
        diff_double_data = np.diff(diff_data, prepend=0)
        
        # Step 1. Create PAA-SAX intervals
        #
        # Raw dataset will be PAA-ed to catch the pattern robust to outliers and noises
        #

        
        SAX_uniform = SAXify(data_transformed,  alphabet = alphabets_1, n_bins = alphabet_size, glob=glob, strategy='uniform')
        SAX_quantile = SAXify(data_transformed,  alphabet = alphabets_2, n_bins = alphabet_size, glob=glob, strategy='quantile')
        SAX_normal = SAXify(data_transformed,  alphabet = alphabets_3, n_bins = alphabet_size, glob=glob, strategy='normal')
        
        PAA_SAX_uniform = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_4, n_bins = alphabet_size, glob=glob, strategy='uniform')
        PAA_SAX_quantile = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_5, n_bins = alphabet_size, glob=glob, strategy='quantile')
        PAA_SAX_normal = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_6, n_bins = alphabet_size, glob=glob, strategy='normal')

        # Step 2. Create only-SAX intervals
        #
        # derivatives will not be PAA-ed since they only have limited numbers
        # For the same reason diff does not have normal layer since it is too strong assumption
        
        # diff does not have normal layer since it is too strong assumption
        diff_SAX_uniform = SAXify(diff_data,  alphabet = alphabets_7, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
        diff_SAX_quantile = SAXify(diff_data,  alphabet = alphabets_8, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')
        
        # double diff does not have normal layer since it is too strong assumption
        diff_double_uniform_SAX = SAXify(diff_double_data, alphabet = alphabets_9, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
        diff_double_quantile_SAX = SAXify(diff_double_data, alphabet = alphabets_10, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')

        SAX_PAA_info[dim] = {"SAX_uniform": SAX_uniform, 
                             "SAX_quantile": SAX_quantile, 
                             "SAX_normal": SAX_normal, 
                             "PAA_SAX_uniform": PAA_SAX_uniform, 
                             "PAA_SAX_quantile": PAA_SAX_quantile, 
                             "PAA_SAX_normal": PAA_SAX_normal, 
                             "diff_SAX_uniform": diff_SAX_uniform, 
                             "diff_SAX_quantile": diff_SAX_quantile, 
                             "diff_double_uniform_SAX": diff_double_uniform_SAX, 
                             "diff_double_quantile_SAX": diff_double_quantile_SAX}
        
        values_original = ["SAX_uniform", "SAX_quantile", "SAX_normal", "PAA_SAX_uniform", "PAA_SAX_quantile", "PAA_SAX_normal"]
        
        ############ technique choosing ###############
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
        
        if reducedFeature == True:
            bin_edges_norm, bin_edges_uni, bin_edges_ed = cal_edges(data_transformed, alphabet_size)

            # print(np.sqrt(np.mean((bin_edges_norm - bin_edges_uni)**2)), np.sqrt(np.mean((bin_edges_norm - bin_edges_ed)**2)))

            X_PAA = PiecewiseAggregateApproximation(window_size=window_size).transform(data_transformed)
            bin_edges_norm_PAA, bin_edges_uni_PAA, bin_edges_ed_PAA = cal_edges(X_PAA, alphabet_size)

            dists = [bin_edges_uni, bin_edges_ed, bin_edges_norm, bin_edges_uni_PAA, bin_edges_ed_PAA, bin_edges_norm_PAA]

            dims = []
            for i, dim in enumerate(dists):
                dims.append([])
                for dim2 in dists:
                    dims[i].append(np.sqrt(np.mean((dim-dim2)**2)))
            
            # print(dims)
            #plt.plot(np.sort(np.array(dims).mean(axis=0))[::-1])
            original_indices = np.argsort(np.array(dims).mean(axis=0))[::-1]
            indices, knee = detect_knee_point(np.sort(np.array(dims).mean(axis=0))[::-1], range(len(dims)))
            indices = original_indices[indices]

            #for values_original we perform bin analysis and choose the ones meeting the requirement
            values[dim] = [values_original[i] for i in indices] # choose the dimensions!
        else:
            values[dim] = values_original

    new_database = []
    
    for row in range(data.shape[0]):
        # Each row, each dimension 
        newRow = []

        for dim in range(dims):
            # We need to add those different intervals in the same database
            newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["diff_SAX_quantile"][row], mask[row][dim], end_size = diff_data.shape[-1], window_size = window_size_slope)
            newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["diff_double_quantile_SAX"][row],mask[row][dim],  end_size = diff_double_data.shape[-1], window_size = window_size_slope)
            newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["diff_SAX_uniform"][row],mask[row][dim],  end_size = diff_data.shape[-1], window_size = window_size_slope)
            newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["diff_double_uniform_SAX"][row],mask[row][dim],  end_size = diff_double_data.shape[-1], window_size = window_size_slope)
            for value in values[dim]:
                if value in ["SAX_uniform", "SAX_quantile", "SAX_normal"]:
                    newRow += createIntervalsMissingInterval(SAX_PAA_info[dim][value][row],mask[row][dim],   end_size = data_transformed.shape[-1], window_size = 1)
                elif value in ["PAA_SAX_uniform", "PAA_SAX_quantile", "PAA_SAX_normal"]:
                    newRow += createIntervals(SAX_PAA_info[dim][value][row], end_size = data_transformed.shape[-1], window_size = window_size)
            # if normal == True:
            #     newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["SAX_normal"][row],mask[row][dim],   end_size = data_transformed.shape[-1], window_size = 1)
            #     newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_normal"][row], end_size = data_transformed.shape[-1], window_size = window_size)
            # if uniform == True:
            #     newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["SAX_uniform"][row], mask[row][dim],  end_size = data_transformed.shape[-1], window_size = 1)
            #     newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_uniform"][row], end_size = data_transformed.shape[-1], window_size = window_size)
            # if quantile == True:
            #     newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["SAX_quantile"][row],mask[row][dim],   end_size = data_transformed.shape[-1], window_size = 1)
            #     newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_quantile"][row], end_size = data_transformed.shape[-1], window_size = window_size)
            
        new_database.append(sorted(newRow))
    return new_database

def createMultivariateDatabase(data, window_size = 10, window_size_slope = 5, alphabet_size = 3, alphabet_size_slope = 5, glob=False, uniform=True, quantile=True, normal=True, varyingLength=False, reducedFeature=False):
    alphabet_origin = 97
    
    # get dimension
    # row, dimension, length
    dims = data.shape[1]
    diff_data = np.zeros(1)
    diff_double_data = np.zeros(1)
    data_transformed = np.zeros(1)

    SAX_PAA_info = {}
    values = {}

    step_size = alphabet_size*6 + alphabet_size_slope*4
    # for each dimension we need to save SAX_PAA information
    for dim in range(dims):
        #transform the data into the understandable form
        
        #data_transformed = np.array([i.to_numpy() for i in data[dim]])
        data_transformed = data[:, dim, :]

        alphabets_1 = [step_size*dim + i for i in range(alphabet_size)] #0,1,2
        alphabets_2 = [step_size*dim + i + alphabet_size for i in range(alphabet_size)] #3,4,5
        alphabets_3 = [step_size*dim + i + alphabet_size*2 for i in range(alphabet_size)] #6,7,8
        alphabets_4 = [step_size*dim + i + alphabet_size*3 for i in range(alphabet_size)] #9,10,11
        alphabets_5 = [step_size*dim + i + alphabet_size*4 for i in range(alphabet_size)] #12,13,14
        alphabets_6 = [step_size*dim + i + alphabet_size*5 for i in range(alphabet_size)] #15,16,17

        # FOR SLOPES
        alphabets_7 = [step_size*dim  + i + alphabet_size*6 for i in range(alphabet_size_slope)] #18, 19
        alphabets_8 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*1 for i in range(alphabet_size_slope)] #20,21
        alphabets_9 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*2 for i in range(alphabet_size_slope)] #22,23
        alphabets_10 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*3 for i in range(alphabet_size_slope)] #24,25

        PAA_transformer = PiecewiseAggregateApproximation(window_size=window_size)

        # diff the dataset
        diff_data = np.diff(data_transformed, prepend=0)
        diff_double_data = np.diff(diff_data, prepend=0)
        
        # Step 1. Create PAA-SAX intervals
        #
        # Raw dataset will be PAA-ed to catch the pattern robust to outliers and noises
        #

        
        SAX_uniform = SAXify(data_transformed,  alphabet = alphabets_1, n_bins = alphabet_size, glob=glob, strategy='uniform')
        SAX_quantile = SAXify(data_transformed,  alphabet = alphabets_2, n_bins = alphabet_size, glob=glob, strategy='quantile')
        SAX_normal = SAXify(data_transformed,  alphabet = alphabets_3, n_bins = alphabet_size, glob=glob, strategy='normal')
        
        PAA_SAX_uniform = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_4, n_bins = alphabet_size, glob=glob, strategy='uniform')
        PAA_SAX_quantile = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_5, n_bins = alphabet_size, glob=glob, strategy='quantile')
        PAA_SAX_normal = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_6, n_bins = alphabet_size, glob=glob, strategy='normal')

        # Step 2. Create only-SAX intervals
        #
        # derivatives will not be PAA-ed since they only have limited numbers
        # For the same reason diff does not have normal layer since it is too strong assumption
        
        # diff does not have normal layer since it is too strong assumption
        diff_SAX_uniform = SAXify(diff_data,  alphabet = alphabets_7, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
        diff_SAX_quantile = SAXify(diff_data,  alphabet = alphabets_8, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')
        
        # double diff does not have normal layer since it is too strong assumption
        diff_double_uniform_SAX = SAXify(diff_double_data, alphabet = alphabets_9, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
        diff_double_quantile_SAX = SAXify(diff_double_data, alphabet = alphabets_10, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')

        SAX_PAA_info[dim] = {"SAX_uniform": SAX_uniform, 
                             "SAX_quantile": SAX_quantile, 
                             "SAX_normal": SAX_normal, 
                             "PAA_SAX_uniform": PAA_SAX_uniform, 
                             "PAA_SAX_quantile": PAA_SAX_quantile, 
                             "PAA_SAX_normal": PAA_SAX_normal, 
                             "diff_SAX_uniform": diff_SAX_uniform, 
                             "diff_SAX_quantile": diff_SAX_quantile, 
                             "diff_double_uniform_SAX": diff_double_uniform_SAX, 
                             "diff_double_quantile_SAX": diff_double_quantile_SAX}
        values_original = ["SAX_uniform", "SAX_quantile", "SAX_normal", "PAA_SAX_uniform", "PAA_SAX_quantile", "PAA_SAX_normal"]
        
        ############ technique choosing ###############
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
        
        if reducedFeature == True:
            bin_edges_norm, bin_edges_uni, bin_edges_ed = cal_edges(data_transformed, alphabet_size)

            # print(np.sqrt(np.mean((bin_edges_norm - bin_edges_uni)**2)), np.sqrt(np.mean((bin_edges_norm - bin_edges_ed)**2)))

            X_PAA = PiecewiseAggregateApproximation(window_size=window_size).transform(data_transformed)
            bin_edges_norm_PAA, bin_edges_uni_PAA, bin_edges_ed_PAA = cal_edges(X_PAA, alphabet_size)

            dists = [bin_edges_uni, bin_edges_ed, bin_edges_norm, bin_edges_uni_PAA, bin_edges_ed_PAA, bin_edges_norm_PAA]

            dimarr = []
            for i, dimen in enumerate(dists):
                dimarr.append([])
                for dim2 in dists:
                    dimarr[i].append(np.sqrt(np.mean((dimen-dim2)**2)))
            
            # print(dims)
            #plt.plot(np.sort(np.array(dims).mean(axis=0))[::-1])
            original_indices = np.argsort(np.array(dimarr).mean(axis=0))[::-1]
            indices, knee = detect_knee_point(np.sort(np.array(dimarr).mean(axis=0))[::-1], range(len(dimarr)))
            # print(indices)
            indices = original_indices[indices]

            #for values_original we perform bin analysis and choose the ones meeting the requirement
            values[dim] = [values_original[i] for i in indices] # choose the dimensions!
        else:
            values[dim] = values_original

    new_database = []
    
    for row in range(data.shape[0]):
        # Each row, each dimension 
        newRow = []

        for dim in range(dims):
            # We need to add those different intervals in the same database
            newRow += createIntervals(SAX_PAA_info[dim]["diff_SAX_quantile"][row], end_size = diff_data.shape[-1], window_size = window_size_slope)
            newRow += createIntervals(SAX_PAA_info[dim]["diff_double_quantile_SAX"][row], end_size = diff_double_data.shape[-1], window_size = window_size_slope)
            newRow += createIntervals(SAX_PAA_info[dim]["diff_SAX_uniform"][row], end_size = diff_data.shape[-1], window_size = window_size_slope)
            newRow += createIntervals(SAX_PAA_info[dim]["diff_double_uniform_SAX"][row], end_size = diff_double_data.shape[-1], window_size = window_size_slope)
            
            for value in values[dim]:
                if value in ["SAX_uniform", "SAX_quantile", "SAX_normal"]:
                    newRow += createIntervals(SAX_PAA_info[dim][value][row],  end_size = data_transformed.shape[-1], window_size = 1)
                elif value in ["PAA_SAX_uniform", "PAA_SAX_quantile", "PAA_SAX_normal"]:
                    newRow += createIntervals(SAX_PAA_info[dim][value][row], end_size = data_transformed.shape[-1], window_size = window_size)
            # if normal == True:
            #     newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["SAX_normal"][row],mask[row][dim],   end_size = data_transformed.shape[-1], window_size = 1)
            #     newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_normal"][row], end_size = data_transformed.shape[-1], window_size = window_size)
            # if uniform == True:
            #     newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["SAX_uniform"][row], mask[row][dim],  end_size = data_transformed.shape[-1], window_size = 1)
            #     newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_uniform"][row], end_size = data_transformed.shape[-1], window_size = window_size)
            # if quantile == True:
            #     newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["SAX_quantile"][row],mask[row][dim],   end_size = data_transformed.shape[-1], window_size = 1)
            #     newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_quantile"][row], end_size = data_transformed.shape[-1], window_size = window_size)
            
        new_database.append(sorted(newRow))
    return new_database

     
def createTrend(data):
    # We are creating a monotonic trend here!
    prevCol = None
    prevGradSign = -9
    count = 0
    prevFinalCount = 0
    gradDict = {False: "-", True: "+", None: "0"}
    newRow = []
    flagZero = False
    
    for currentCol in data: 
        # first column setting
        if prevCol == None:
            count += 1
        # from second column, we can get the gradient
        else:
            # we only check the sign of the gradient
            grad = currentCol - prevCol
            
            # Define currentGradSign (None, True, False)
            if grad == 0:
                currentGradSign = None
            elif grad > 0:
                currentGradSign = True
            else:
                currentGradSign = False
                
            # When we do not have any sign (meaning the first column)
            if prevGradSign == -9:
                count += 1
                
            # if it is the second column, we need to keep it and continue
            else:
                # check whether current sign is the same or not
                if (prevGradSign == currentGradSign):
                    #if they are the same, just continue with the count
                    count += 1
                # when the sign is different -> means zero or opposite T/F
                else:
                    #We put this new one only if the current sign was not zero
                    #if flagZero == False:
                        #If not, we need to put a new trend 
                    newRow.append((gradDict[prevGradSign], prevFinalCount, prevFinalCount + count))
                    #else:
                    #    flagZero = False
                    prevFinalCount = prevFinalCount + count + 1
                    count = 0
            
            # Regardless of the column order, if it is first time to see zero:
            if currentGradSign == None and flagZero == False:
                flagZero = True
            
            # Update previous grad sign
            prevGradSign = currentGradSign
            
            
        prevCol = currentCol
        
    #print((gradDict[prevGradSign], prevFinalCount, prevFinalCount + count))
    #if flagZero == False:
    newRow.append((gradDict[prevGradSign], prevFinalCount, prevFinalCount + count))
    return newRow
                        
def SAXify(data, alphabet,  n_bins = 5, glob=False, strategy='uniform'):
    sax = SymbolicAggregateApproximation(n_bins = n_bins, alphabet = alphabet, strategy=strategy)
    
    # global option makes SAX range with all data points
    if glob == True:
        globalvals = np.concatenate(data)
        data_new = sax.fit_transform([globalvals])
        data_new = data_new.reshape(data.shape)
    else:
        data_new = sax.fit_transform(data)
    return data_new

def removeSpecificEventLabel(data, labels):
    newData = []
    for row in data:
        if row[0] not in labels:
            newData.append(row)
    return newData

def checkLabelBalance(labels):
    return np.unique(labels, return_counts = True)[1] / len(labels)

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

def chooseDimensionsHistogram(X_train, X_test, n_bins = 5):
    X_new = []

    sax = SymbolicAggregateApproximation(n_bins = n_bins, strategy='normal')

    for dim in range(X_train.shape[1]):
        #transform the data into the understandable form
        
        #data_transformed = np.array([i.to_numpy() for i in data[dim]])
        data_transformed = X_train[:, dim, :]
        X_train_1 = sax.fit_transform(data_transformed)
        X_new.append(X_train_1)

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
    indices, knee = detect_knee_point(np.array(dims).mean(axis=0), range(X_new_final.shape[1]))
    X_train = X_train[:,indices,:]
    X_test = X_test[:,indices,:]
    
    return (X_train, X_test)

def trialSequence_multivariate(X_train, X_test, y_train, y_test, window_size = 10, window_size_slope = 5, 
                    alphabet_size = 5, alphabet_size_slope = 1, 
                    glob=False, step=0, split_no = 1, varyingLength = False, missingDataTest = False,
                    mask_train = False, mask_test = False, dim_selection = False, reducedFeature=False):

    t1 = time.perf_counter()
    total_dims = []
    vectorizer = DictVectorizer(dtype=np.uint16, sparse=True)
    


    # size, dimension, length
    if varyingLength == True:
        # row, split, dimension, length -> split, row, dimension, length
        X_train_split = [np.array_split(i, split_no, axis=-1) for i in X_train]
        X_test_split = [np.array_split(i, split_no, axis=-1) for i in X_test]
        X_train_split = [[row[split] for row in X_train_split] for split in range(split_no)]
        X_test_split = [[row[split] for row in X_test_split] for split in range(split_no)]

        if len(X_train_split[0][0][0]) < window_size:
            # NO ERROR MSG IN DEBUG MODE
            # print("WINDOW SIZE IS SMALLER THAN SPLIT")
            return None, None
    else:
        X_train_split = np.array_split(X_train, split_no, axis=-1)
        X_test_split = np.array_split(X_test, split_no, axis=-1)
        if X_train_split[-1].shape[-1] < window_size:
            # NO ERROR MSG IN DEBUG MODE
            print(X_train_split[-1].shape)
            print("WINDOW SIZE IS SMALLER THAN SPLIT")
            return None, None

    trainset = []
    testset = []      
    
    for split_part in zip(X_train_split, X_test_split):

        if dim_selection == "ECP":
            cs = ElbowClassPairwise()
            cs.fit(split_part[0], y_train)
            split_part = (cs.transform(split_part[0]), cs.transform(split_part[1]))
        elif dim_selection == "ECS":
            cs = ElbowClassSum()
            cs.fit(split_part[0], y_train)
            split_part = (cs.transform(split_part[0]), cs.transform(split_part[1]))
        elif dim_selection == "HIST":
            split_part = chooseDimensionsHistogram(split_part[0], split_part[1])

        total_dims.append(split_part[0].shape[1])

        # print("conversion start")
        if varyingLength == True:
            train_dataset = createMultivariateDatabaseVL(split_part[0], window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope= alphabet_size_slope, glob=glob, reducedFeature=reducedFeature)
            test_dataset = createMultivariateDatabaseVL(split_part[1], window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope = alphabet_size_slope, glob=glob, reducedFeature=reducedFeature)

        else:
            if missingDataTest == False:
                train_dataset = createMultivariateDatabase(split_part[0], window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope= alphabet_size_slope, glob=glob, reducedFeature=reducedFeature)
                test_dataset = createMultivariateDatabase(split_part[1], window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope = alphabet_size_slope, glob=glob, reducedFeature=reducedFeature)
            else:
                train_dataset = createMultivariateDatabaseMissingData(split_part[0], mask_train, window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope= alphabet_size_slope, glob=glob)
                test_dataset = createMultivariateDatabaseMissingData(split_part[1], mask_test, window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope = alphabet_size_slope, glob=glob)
            
        # row, dimension, length
        
        # print("conversion done")
        algorithm = ZTime.ZTime(step=step)
        tsMatrix = algorithm.train(train_dataset, y_train)
        # row_labels = list(trainset) 
        #matrix = vectorizer.fit_transform([data[i] for i in row_labels]) 
        trainset_tmp = vectorizer.fit_transform([tsMatrix[i] for i in list(tsMatrix)]).T
        trainset.append(trainset_tmp)
        # print("Training done")
        # print(trainset.shape, y_train.shape)
        # Testing process
        tsMatrix_test = algorithm.test(test_dataset)
        #testset = pd.DataFrame.from_dict(tsMatrix_test).fillna(0).to_numpy()
        # testset = convert(tsMatrix_test)
        #matrix = vectorizer.fit_transform([data[i] for i in row_labels]) 
        testset_tmp = vectorizer.fit_transform([tsMatrix_test[i] for i in list(tsMatrix_test)]).T
        testset.append(testset_tmp)

    # Memory efficiency
    algorithm = None
    # tsMatrix = None
    # tsMatrix_test = None
    
    trainset = hstack(trainset)
    testset = hstack(testset)

    # print(sparse_memory_usage(trainset), getsizeof(trainset))
    
    #normalization for logistic regression
    # scaler = StandardScaler().fit(trainset)
    # trainset = scaler.transform(trainset)
    # testset = scaler.fit_transform(testset)
    #print(train_norm.shape, test_norm.shape)

    # apply logistic regresiion
    classifier = RidgeClassifierCV(normalize = True)
    classifier.fit(trainset, y_train)
    ############ Interpretability test ################
    # features = list(tsMatrix.keys())
    # coeff = np.array(classifier.coef_)
    # ind = np.argpartition(coeff[1], -10)[-10:]
    # for i in ind:
    #     print(i)
    #     print(features[i])
    ###########################################
    # trainset = None
    lr_score = classifier.score(testset, y_test)
    t2 = time.perf_counter()
    timegap = t2-t1
    return trainset.shape[1], np.mean(total_dims), lr_score, timegap

def trialSequence_multivariate_old(X_train, X_test, y_train, y_test, window_size = 10, window_size_slope = 5, 
                    alphabet_size = 5, alphabet_size_slope = 1, 
                    glob=False, step=0, split_no = 1):

    t1 = time.perf_counter()

    # size, dimension, length
    X_train_split = np.array_split(X_train, split_no, axis=-1)
    X_test_split = np.array_split(X_test, split_no, axis=-1)

    trainset_created = []
    testset_created = []

    if X_train_split[-1].shape[1] < window_size:
        # NO ERROR MSG IN DEBUG MODE
        #print("WINDOW SIZE IS SMALLER THAN SPLIT")
        return None, None
    
    for split_part in zip(X_train_split, X_test_split):

        train_dataset = createMultivariateDatabase(split_part[0], window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope= alphabet_size_slope, glob=glob)
        test_dataset = createMultivariateDatabase(split_part[1], window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope = alphabet_size_slope, glob=glob)

        # Training process
        algorithm = ZTime.ZTime(step)
        # print(train_dataset)
        tsMatrix = algorithm.train(train_dataset, y_train)
        trainset = pd.DataFrame.from_dict(tsMatrix).fillna(0).to_numpy(dtype=np.uint8)
        trainset_created.append(trainset)
        # print("Training done")
        # print(trainset.shape, y_train.shape)
        # Testing process
        tsMatrix_test = algorithm.test(test_dataset)
        testset = pd.DataFrame.from_dict(tsMatrix_test).fillna(0).to_numpy(dtype=np.uint8)
        testset_created.append(testset)
        # print("Testing done")
        # print(testset.shape, y_test.shape)

    trainset = np.concatenate(trainset_created, axis=1).astype(np.uint8)
    testset = np.concatenate(testset_created, axis=1).astype(np.uint8)

    #normalization for logistic regression
    scaler = StandardScaler().fit(trainset)
    trainset = scaler.transform(trainset)
    testset = scaler.fit_transform(testset)
    #print(train_norm.shape, test_norm.shape)

    # apply logistic regresiion
    classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    classifier.fit(trainset, y_train)
    lr_score = classifier.score(testset, y_test)
    t2 = time.perf_counter()
    timegap = t2-t1
    return lr_score, timegap

def trialSequence(X_train, X_test, y_train, y_test, 
                    window_size = 10, window_size_slope = 5, 
                    alphabet_size = 5, alphabet_size_slope = 5, 
                    glob=False, step=10, split_no=2, debug=False):

    t1 = time.perf_counter()

    X_train_split = np.array_split(X_train, split_no, axis=1)
    X_test_split = np.array_split(X_test, split_no, axis=1)

    trainset_created = []
    testset_created = []

    if X_train_split[-1].shape[1] < window_size:
        if debug == True:
            print("WINDOW SIZE IS SMALLER THAN SPLIT")  
        return None, None
    
    
    if debug == True:
        t2 = time.perf_counter()
        print("1. Array split time:", t2-t1)    

    for split_part in zip(X_train_split, X_test_split):
        
        
        ts1 = time.perf_counter()
        #setting the variables
        train_dataset = createDatabase(split_part[0], window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope= alphabet_size_slope, glob=glob)
        test_dataset = createDatabase(split_part[1], window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope= alphabet_size_slope, glob=glob)
        
        
        if debug == True:
            ts2 =  time.perf_counter()
            print("### Database part:", ts2-ts1)
        # constraint setting - minSupPercent, maxSupPercent, gap, level (mandatory), database
        # WE CHOOSE THE FEATURES FROM THE TRAINING SET BASED ON ITS MINSUP/MAXSUP BUT ON TESTSET WE IGNORE
        
        
        algorithm = ZTime.ZTime(step=step)
        tsMatrix = algorithm.train(train_dataset, y_train)
        
        if debug == True:
            ts3 =  time.perf_counter()
            print("### ZTIME training part:", ts3-ts2)
        #trainset = pd.DataFrame.from_dict(tsMatrix).fillna(0).to_numpy()
        #trainset = convert(tsMatrix)
        vectorizer = DictVectorizer(dtype=np.uint16, sparse=True)
        # row_labels = list(trainset) 
        row_labels = list(tsMatrix) 
        #matrix = vectorizer.fit_transform([data[i] for i in row_labels]) 
        trainset = vectorizer.fit_transform([tsMatrix[i] for i in row_labels]).T
        # print(trainset.shape)
        trainset_created.append(trainset)
        
        if debug == True:
            ts4 =  time.perf_counter()
            print("### PD from dict training part:", ts4-ts3)
        # print("Training done")
        # print(trainset.shape, y_train.shape)
        # Testing process
        tsMatrix_test = algorithm.test(test_dataset)
        if debug == True:
            ts5 = time.perf_counter()
            print("### ZTIME test part:", ts5-ts4)
        #testset = pd.DataFrame.from_dict(tsMatrix_test).fillna(0).to_numpy()
        # testset = convert(tsMatrix_test)
        row_labels = list(tsMatrix_test) 
        #matrix = vectorizer.fit_transform([data[i] for i in row_labels]) 
        testset = vectorizer.fit_transform([tsMatrix_test[i] for i in row_labels]).T
        testset_created.append(testset)

        if debug == True:   
            ts6 =  time.perf_counter()
            print("### PD from dict training part:", ts6-ts5)

        # print("Testing done")
        # print(testset.shape, y_test.shape)
    
    if debug == True:       
        t3 = time.perf_counter()
        print("2. Creation time:", t3-t2)
    # Memory fix
    #trainset = np.concatenate(trainset_created, axis=1).astype(np.uint8)
    #testset = np.concatenate(testset_created, axis=1).astype(np.uint8)
    trainset = hstack(trainset_created)
    testset = hstack(testset_created)
    # print(sparse_memory_usage(trainset), getsizeof(trainset))

    if debug == True:        
        t4 = time.perf_counter()
        print("3. Concatenate time:", t4-t3)
    #normalization for logistic regression
    # Maybe pipeline can be better for memory fix
    #scaler = StandardScaler().fit(trainset)
    # Memory allocation fix
    #trainset = scaler.transform(trainset)
    #testset = scaler.transform(testset)
    #print(train_norm.shape, test_norm.shape)

    if debug == True:
        t5 = time.perf_counter()
        print("4. Scaling time:", t5-t4)

    classifier3 = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    classifier3.fit(trainset, y_train)
    
    # transform test set and predict
    lr_score_3 = classifier3.score(testset, y_test)
    t6 = time.perf_counter()
    
    if debug == True:
        print("5. Training and test:", t6-t5)
        print("Total:", t6-t1)
    timegap = t6-t1
    # print(f"{cycle},{step},{alphabet_size},{alphabet_size_slope},{window_size},{split_no},{lr_score:.2%},{lr_score_2:.2%},{lr_score_3:.2%},{timegap:.2f}")
    
    return lr_score_3, timegap
    
def trialSequence_old(X_train, X_test, y_train, y_test, 
                    window_size = 10, window_size_slope = 5, 
                    alphabet_size = 5, alphabet_size_slope = 5, 
                    glob=False, step=10, split_no=1):

    t1 = time.perf_counter()

    X_train_split = np.array_split(X_train, split_no, axis=1)
    X_test_split = np.array_split(X_test, split_no, axis=1)

    trainset_created = []
    testset_created = []

    if X_train_split[-1].shape[1] < window_size:
        # print("WINDOW SIZE IS SMALLER THAN SPLIT")
        return None, None
    
    for split_part in zip(X_train_split, X_test_split):
        
        # t1 = time.perf_counter()
        #setting the variables
        train_dataset = createDatabase(split_part[0], window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope= alphabet_size_slope, glob=glob)
        test_dataset = createDatabase(split_part[1], window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope= alphabet_size_slope, glob=glob)
        
        # constraint setting - minSupPercent, maxSupPercent, gap, level (mandatory), database
        # WE CHOOSE THE FEATURES FROM THE TRAINING SET BASED ON ITS MINSUP/MAXSUP BUT ON TESTSET WE IGNORE

        algorithm = ZTime.ZTime(step=step)
        tsMatrix = algorithm.train(train_dataset, y_train)
        trainset = pd.DataFrame.from_dict(tsMatrix).fillna(0).to_numpy()
        trainset_created.append(trainset)
        # print("Training done")
        # print(trainset.shape, y_train.shape)
        # Testing process
        tsMatrix_test = algorithm.test(test_dataset)
        testset = pd.DataFrame.from_dict(tsMatrix_test).fillna(0).to_numpy()
        testset_created.append(testset)
        # print("Testing done")
        # print(testset.shape, y_test.shape)
    
    # Memory fix
    trainset = np.concatenate(trainset_created, axis=1).astype(np.uint8)
    testset = np.concatenate(testset_created, axis=1).astype(np.uint8)

    #normalization for logistic regression
    # Maybe pipeline can be better for memory fix
    #scaler = StandardScaler().fit(trainset)
    # Memory allocation fix
    #trainset = scaler.transform(trainset)
    #testset = scaler.transform(testset)
    #print(train_norm.shape, test_norm.shape)

    # apply logistic regresiion
    # classifier1 = LogisticRegression()
    # classifier1.fit(train_norm, y_train)
    # lr_score = classifier1.score(test_norm, y_test)
    
    # classifier2 =  SGDClassifier(max_iter=1000, tol=1e-3, penalty='elasticnet', n_jobs=-1)
    # classifier2.fit(train_norm, y_train)
    # lr_score_2 = classifier2.score(test_norm, y_test)

    # classifier3 = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    # classifier3.fit(trainset, y_train)

    # transform test set and predict
    lr_score_3 = classifier3.score(testset, y_test)

    t2 = time.perf_counter()
    timegap = t2-t1
    # print(f"{cycle},{step},{alphabet_size},{alphabet_size_slope},{window_size},{split_no},{lr_score:.2%},{lr_score_2:.2%},{lr_score_3:.2%},{timegap:.2f}")
    
    return lr_score_3, timegap

def simpleTrial(data, alphabet_size = 5, alphabet_size_slope = 5, window_size = 10, split_no = 1, step=10, multivariate=False, varyingLength=False, dim_selection = False, reducedFeature=False):
    X_train = np.nan_to_num(znorm(np.nan_to_num(data['TRAIN']['X'], 0)), 0)
    y_train = data["TRAIN"]["y"]
    X_test = np.nan_to_num(znorm(np.nan_to_num(data['TEST']['X'], 0)), 0)
    y_test = data["TEST"]["y"]

    if multivariate == True:
        feature_no, total_dims, lr_score, timegap = trialSequence_multivariate(X_train, X_test, y_train, y_test, window_size = window_size, 
            window_size_slope = 1, alphabet_size = alphabet_size, alphabet_size_slope=alphabet_size_slope, step=step, split_no = split_no, varyingLength=varyingLength,
            dim_selection = dim_selection, reducedFeature=reducedFeature)
    else:    
        feature_no, lr_score, timegap = trialSequence(X_train, X_test, y_train, y_test, window_size = window_size, 
            window_size_slope = 1, alphabet_size = alphabet_size, alphabet_size_slope=alphabet_size_slope,split_no = split_no, step=step)

    
    return feature_no, total_dims, lr_score, timegap

    
def randomizedSearchKFold(data, steps, alphabet_sizes, alphabet_sizes_slope, window_sizes, split_nos, k=3, cycle = 100):
    X_train = znorm(np.nan_to_num(data['TRAIN']['X'], 0))
    y_train = data["TRAIN"]["y"]
    X_test = znorm(np.nan_to_num(data['TEST']['X'], 0))
    y_test = data["TEST"]["y"]

    best_params = []
    best_acc = -np.inf
    
    chosen_steps = np.random.choice(steps, cycle)
    chosen_alphabet_sizes = np.random.choice(alphabet_sizes, cycle)
    chosen_alphabet_sizes_slope = np.random.choice(alphabet_sizes_slope, cycle)
    chosen_window_sizes = np.random.choice(window_sizes, cycle)
    chosen_split_nos = np.random.choice(split_nos, cycle)

    #X_train_rs, X_val_rs, y_train_rs, y_test_rs = train_test_split(X_train, y_train, test_size=0.30, stratify=y_train)

    kf = StratifiedKFold(n_splits=k)

    # validation
    count = 1
    
    #split_values = kf.split(X_train, y_train)

    for step, window_size, alphabet_size, alphabet_size_slope, split_no in zip(chosen_steps, chosen_window_sizes, chosen_alphabet_sizes, chosen_alphabet_sizes_slope, chosen_split_nos):
        scores = []
        
        for train_index, test_index in kf.split(X_train, y_train):
            X_train_rs, X_val_rs = X_train[train_index], X_train[test_index]
            y_train_rs, y_val_rs = y_train[train_index], y_train[test_index]
        
            lr_score, timegap = trialSequence(X_train_rs, X_val_rs, y_train_rs, y_val_rs, window_size = window_size, 
                        window_size_slope = 1, alphabet_size = alphabet_size, alphabet_size_slope=alphabet_size_slope, step=step, split_no = split_no)
            scores.append(lr_score)
        
        if np.mean(scores) > best_acc:
            best_acc = np.mean(scores)
            best_params = [window_size, 1, alphabet_size, alphabet_size_slope, split_no, step]
    count += 1
    
    # real
    lr_score, timegap = trialSequence(X_train, X_test, y_train, y_test, window_size = best_params[0], 
        window_size_slope = best_params[1], alphabet_size = best_params[2], alphabet_size_slope=best_params[3], split_no = best_params[4], step=best_params[5])

    print(f"Final result: {lr_score}")



def randomizedSearch(data, steps, alphabet_sizes, alphabet_sizes_slope, window_sizes, split_nos, cycle = 20, multivariate=False, varyingLength=False):
    if varyingLength == False:
        X_train = np.nan_to_num(znorm(np.nan_to_num(data['TRAIN']['X'], 0)), 0)
        X_test = np.nan_to_num(znorm(np.nan_to_num(data['TEST']['X'], 0)),0)
    else:
        X_train = [znorm(np.array([i.values for i in data["TRAIN"]["X"].iloc[j]])) for j in range(data["TRAIN"]["X"].shape[0])]
        X_test = [znorm(np.array([i.values for i in data["TEST"]["X"].iloc[j]])) for j in range(data["TEST"]["X"].shape[0])]

    y_train = data["TRAIN"]["y"]
    y_test = data["TEST"]["y"]

    #print(np.isnan(X_train).sum())
    best_params = []
    best_acc = -1
    lr_score = 0

    chosen_steps = np.random.choice(steps, cycle)
    chosen_alphabet_sizes = np.random.choice(alphabet_sizes, cycle)
    chosen_alphabet_sizes_slope = np.random.choice(alphabet_sizes_slope, cycle)
    chosen_window_sizes = np.random.choice(window_sizes, cycle)
    chosen_split_nos = np.random.choice(split_nos, cycle)
    try:
        X_train_rs, X_val_rs, y_train_rs, y_val_rs = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train)
    except:
        # Only when there is not enough class member (<2)
        X_train_rs, X_val_rs, y_train_rs, y_val_rs = train_test_split(X_train, y_train, test_size=0.33)

    count = 0
    
    #for step, window_size, alphabet_size, alphabet_size_slope, split_no in zip(chosen_steps, chosen_window_sizes, chosen_alphabet_sizes, chosen_alphabet_sizes_slope, chosen_split_nos):
    while True:
        step = np.random.choice(steps, 1)[0]
        alphabet_size = np.random.choice(alphabet_sizes, 1)[0]
        alphabet_size_slope = np.random.choice(alphabet_sizes_slope, 1)[0]
        window_size = np.random.choice(window_sizes, 1)[0]
        split_no = np.random.choice(split_nos, 1)[0]

        # print("getting in")
        try:
            if multivariate:
                lr_score, timegap = trialSequence_multivariate(X_train_rs, X_val_rs, y_train_rs, y_val_rs, window_size = window_size, 
                            window_size_slope = 1, alphabet_size = alphabet_size, alphabet_size_slope=alphabet_size_slope, step=step, split_no = split_no, varyingLength=varyingLength)
            else:
                lr_score, timegap = trialSequence(X_train_rs, X_val_rs, y_train_rs, y_val_rs, window_size = window_size, 
                            window_size_slope = 1, alphabet_size = alphabet_size, alphabet_size_slope=alphabet_size_slope, step=step, split_no = split_no)
        except:
            print("TrialSequence Failed")
            continue
        
        print(lr_score)
        if lr_score == None:
            continue

        if lr_score > best_acc:
            best_acc = lr_score
            best_params = [window_size, 1, alphabet_size, alphabet_size_slope, split_no, step]
            print(best_acc, best_params)
        count += 1

        if count == cycle:
            break
    
    # print("Finished the process")
    # real
    if multivariate:
        lr_score, timegap = trialSequence_multivariate(X_train, X_test, y_train, y_test, window_size = best_params[0], 
            window_size_slope = best_params[1], alphabet_size = best_params[2], alphabet_size_slope=best_params[3], split_no = best_params[4], step=best_params[5], varyingLength=varyingLength)
    else:
        lr_score, timegap = trialSequence(X_train, X_test, y_train, y_test, window_size = best_params[0], 
            window_size_slope = best_params[1], alphabet_size = best_params[2], alphabet_size_slope=best_params[3], split_no = best_params[4], step=best_params[5])

    print(f"Final result: {lr_score}")
    return lr_score, timegap
