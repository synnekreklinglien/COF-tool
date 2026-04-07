# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 13:11:56 2018

@author: hwaln
"""

import csv
import pandas as pd
import os

def getTreaCsvMetaData(fname):
    """
    Read metadata from Treasure standard file

    Parameters
    ----------
    fname : string
        filename for datafile to read

    Returns
    -------
    d : dict
        Dictionary with metadata

    """
    d={}
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        row = next(reader)
        d[row[0]]=row[1]
        nrow = int(row[1])
        for i, row in enumerate(reader):
            if i>(nrow-1):
                break
            if not row:
                break
            #k, v = row
            #d[k] = v
            d[row[0]]=row[1]
        return d

def getTreaCsvTimeSeries(fname, meta):
    """
    Get time sereies data from Treasure standard file

    Parameters
    ----------
    fname : string
        filename for datafile to read
    meta : dict
        Dictionary with metadata

    Returns
    -------
    df : Pandas dataframe
        Dataframe with time series

    """
    hrow = int(meta[list(meta)[0]])
    try:
        df = pd.read_csv(fname, index_col=0, skiprows=hrow-1, sep=';')
    except:
        df = pd.read_csv(fname, index_col=0, skiprows=hrow-1, sep=';', encoding = 'ISO-8859-1')
    try:
        if 'timestamp_format' in meta.keys():
            df.index = pd.to_datetime(df.index, format=meta['timestamp_format'])
        else:
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S%z")
    except Exception as e:
        print(f"Error converting index to datetime: {e}")
        df.index = pd.to_datetime(df.index, errors='coerce')  # Fallback to auto-detection
    else:
        try:
            df.index=pd.to_datetime(df.index, format='%d.%m.%Y %H:%M')
        except ValueError:
            print('Warning: timestamp format not standard. Trying day first')
            df.index=pd.to_datetime(df.index, format='%Y.%m.%d %H:%M')
    if 'time_zone' in meta.keys():
        df.index=df.index.tz_convert(meta['time_zone'])
        if not meta['time_zone']=='Etc/Gmt-1':
            print(f'Warning: {fname} time_zone not standard. Check')
    else:
        print(f'Warning: {fname} time_zone not defined. Check')
    return df

def getTreaCsv(fname):
    """
    get metadata and timeseries from Treasure datafile

    Parameters
    ----------
    fname : string
        filename for datafile to read

    Returns
    -------
    d : dict
        Dictionary with metadata
    df : Pandas dataframe
        Dataframe with time series
    """
    d = getTreaCsvMetaData(fname)
    df = getTreaCsvTimeSeries(fname,d)
    return d, df

def readAllcsvsFromFolder(directory, ext='txt'):
    """
    Read all Treasure datafiles from a specified directory

    Parameters
    ----------
    directory : string
        path to directory
    ext : string, optional
        file extension for datafiles. The default is 'txt'.

    Returns
    -------
    Mdict : dict
        dictionary with metadata dictionaries for each file.
    Bdict : dict
        dictionary with timesereis for each file.

    """
    Mdict = {}
    Bdict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.'+ext):
            print('reading file: '+ filename)
            key=filename.split('.')[0]
            path=directory if not directory=='.' else ''
            Mdict[key], Bdict[key]=getTreaCsv(path+filename)
    return Mdict, Bdict
