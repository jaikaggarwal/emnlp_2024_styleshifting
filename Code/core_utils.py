# import emoji
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
from scipy import stats
from functools import reduce
from itertools import product
from collections import Counter, defaultdict
import sys
import time
from tqdm import tqdm
from constant import *


tqdm.pandas()
pd.options.mode.chained_assignment = None

HTTP_PATTERN = re.compile(r'[\(\[]?https?:\/\/.*?(\s|\Z)[\r\n]*')


class Serialization:
    @staticmethod
    def save_obj(obj, name):
        """
        serialization of an object
        :param obj: object to serialize
        :param name: file name to store the object
        """
        with open(SERIALIZATION_DIR + name + '.pkl', 'wb') as fout:
            pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)
        # end with
    # end def

    @staticmethod
    def load_obj(name):
        """
        de-serialization of an object
        :param name: file name to load the object from
        """
        with open(SERIALIZATION_DIR + name + '.pkl', 'rb') as fout:
            return pickle.load(fout)

def set_intersection(l1, l2):
    """
    Returns the intersection of two lists.
    """
    return list(set(l1).intersection(set(l2)))


def set_union(l1, l2):
    """
    Returns the union of two lists.
    """
    return list(set(l1).union(set(l2)))

def set_difference(l1, l2):
    """
    Returns the difference of two lists.
    """
    return list(set(l1).difference(set(l2)))

def intersect_overlap(l1, l2):
    """
    Returns the intersection of two lists,
    while also describing the size of each list
    and the size of their intersection.
    """
    intersected = set_intersection(l1, l2)
    print(f"Sets of size {len(set(l1))} and {len(set(l2))} have overlap of {len(intersected)}.\n")
    return intersected

def jaccard_similarity(l1, l2):
    """
    Compute the jaccard similarity between two input lists.
    """
    l1 = set(l1)
    l2 = set(l2)
    return np.round(len(l1.intersection(l2)) / len(l1.union(l2)), 2)

def flatten_logic(arr):
    """
    Flattens a nested array. 
    """
    for i in arr:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def create_new_directory(dir_path):
    if not os.path.exists(dir_path):
        print(f"Making new directory at {dir_path}")
        os.makedirs(dir_path)
    else:
        print(f"{dir_path} already exists")
    

def groupby_threshold(df, operation, is_index_level, groupby_column, threshold_column, threshold, to_print):
    
   
    if operation == "count":
        agg_func = lambda x: x.count()
    elif operation == "sum":
        agg_func = lambda x: x.sum()
    elif operation == "mean":
        agg_func = lambda x: x.mean()
    elif operation == "nunique":
        agg_func = lambda x: x.nunique()
    elif operation == "cumsum":
        agg_func = lambda x: x.cumsum()
    elif operation == "min":
        agg_func = lambda x: x.min()
    elif operation == "max":
        agg_func = lambda x: x.max()
    else:
        raise Exception("Please specify the groupby operation: {count, sum, mean, nunique, cumsum, min, max}")
    

    if is_index_level:
        agg = agg_func(df.groupby(level=groupby_column))
        met_cutoff = agg[agg[threshold_column] >= threshold].index.tolist()
        new_df = df[df.index.get_level_values(groupby_column).isin(met_cutoff)]
    else:
        agg = agg_func(df.groupby(by=groupby_column))
        met_cutoff = agg[agg[threshold_column] >= threshold].index.tolist()
        new_df = df[df[groupby_column].isin(met_cutoff)]

    if to_print:
        print(f"Data size before applying threshold to {threshold_column}: {df.shape[0]}")
        print(f"Data size after applying threshold to {threshold_column}: {new_df.shape[0]}")
    return met_cutoff, new_df


def directory_to_df(dir_name, files, file_type, filter=lambda x: x):
    """
    Filter is a lambda function
    """
    if file_type == "csv":
        data_loader = lambda x: pd.read_csv(x)
    elif file_type == "json":
        data_loader = lambda x: pd.read_json(x, lines=True)
    else:
        raise Exception("please specify file type")

    data = []
    for file in tqdm(files):
        curr = data_loader(dir_name + file)
        data.append(filter(curr))
    total = pd.concat(data)
    return total


def filter_df(df, filter_col, filter_val):
    if type(filter_val) != list:
        filter_val = [filter_val]
    return df[df[filter_col].str.lower().isin(filter_val)]

def flatten(arr):
    """
    Wrapper for the generator returned by flatten logic.
    """
    return list(flatten_logic(arr))

def preprocess(text):
    """
    Preprocesses text from Reddit posts and comments.
    """
    # Replace links with LINK token
    line = HTTP_PATTERN.sub(" LINK ", text)
    # Replace irregular symbol with whitespace
    line = re.sub("&amp;#x200b", " ", line)
    # Replace instances of users quoting previous posts with empty string
    line = re.sub(r"&gt;.*?(\n|\s\s|\Z)", " ", line)
    # Replace extraneous parentheses with whitespace
    line = re.sub(r'\s\(\s', " ", line)
    line = re.sub(r'\s\)\s', " ", line)
    # Replace newlines with whitespace
    line = re.sub(r"\r", " ", line)
    line = re.sub(r"\n", " ", line)
    # Replace mentions of users with USER tokens
    line = re.sub("\s/?u/[a-zA-Z0-9-_]*(\s|\Z)", " USER ", line)
    # Replace mentions of subreddits with REDDIT tokens
    line = re.sub("\s/?r/[a-zA-Z0-9-_]*(\s|\Z)", " REDDIT ", line)
    # Replace malformed quotation marks and apostrophes
    line = re.sub("’", "'", line)
    line = re.sub("”", '"', line)
    line = re.sub("“", '"', line)
    # Get rid of asterisks indicating bolded or italicized comments
    line = re.sub("\*{1,}(.+?)\*{1,}", r"\1", line)    
    # Replace emojis with EMOJI token
    # line = emoji.get_emoji_regexp().sub(" EMOJI ", line)
    # Replace all multi-whitespace characters with a single space.
    line = re.sub("\s{2,}", " ", line)
    return line

