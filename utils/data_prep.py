### Data Prep SF Crime Classification ###

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

### Global Constants #######
days_of_week = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6 }
districts = {'RICHMOND': 6, 'CENTRAL': 1, 'NORTHERN': 4, 'TARAVAL': 8, 'BAYVIEW': 0, 'INGLESIDE': 2,
             'PARK': 5, 'MISSION': 3, 'TENDERLOIN': 9, 'SOUTHERN': 7}
labels = {'RECOVERED VEHICLE': 24, 'SUICIDE': 31, 'FRAUD': 13, 'WEAPON LAWS': 38, 'ROBBERY': 25,
          'ARSON': 0, 'SECONDARY CODES': 27, 'SEX OFFENSES FORCIBLE': 28, 'WARRANTS': 37,
          'PROSTITUTION': 23, 'DRUG/NARCOTIC': 7, 'EMBEZZLEMENT': 9, 'TRESPASS': 34, 'LOITERING': 18,
          'KIDNAPPING': 15, 'DRIVING UNDER THE INFLUENCE': 6, 'LARCENY/THEFT': 16, 'VANDALISM': 35,
          'NON-CRIMINAL': 20, 'BURGLARY': 4, 'BAD CHECKS': 2, 'STOLEN PROPERTY': 30, 'EXTORTION': 10,
          'SUSPICIOUS OCC': 32, 'PORNOGRAPHY/OBSCENE MAT': 22, 'LIQUOR LAWS': 17, 'FAMILY OFFENSES': 11,
          'SEX OFFENSES NON FORCIBLE': 29, 'TREA': 33, 'GAMBLING': 14, 'BRIBERY': 3, 'VEHICLE THEFT': 36,
          'FORGERY/COUNTERFEITING': 12, 'ASSAULT': 1, 'DRUNKENNESS': 8, 'MISSING PERSON': 19,
          'DISORDERLY CONDUCT': 5, 'OTHER OFFENSES': 21, 'RUNAWAY': 26}

############################

# Global Variables

street_dict = {}
street_count = 1

# create global data structure:
# columns: year, month, day, hour, minute, dayofweek, district, x-coord, y-coord
# output_component [0, 0, 0, 0, 0, 0, 0, 0.0, 0.0]


def df_read(filename):
    data = pd.read_csv(filename, low_memory=False)
    return data

def dayofweek_encode(data):
    '''
    Encode day of the week as number.
    :param data: dataframe to be processed
    :return: dataframe with DayOfWeek column converted to integers
    '''
    data['DayOfWeek'] = data['DayOfWeek'].apply(lambda x: days_of_week[x])
    return data

def category_encode(data):
    '''
    Encode labels as numbers.
    :param data: dataframe to be processed
    :return: dataframe with Category column converted to integers
    '''
    data['Category'] = data['Category'].apply(lambda x: labels[x])
    return data

def district_encode(data):
    '''
    Encode districts as numbers.
    :param data: dataframe to be processed
    :return: dataframe with PdDistrict column converted to integers
    '''
    data['PdDistrict'] = data['PdDistrict'].apply(lambda x: districts[x])
    return data

def date_expansion(data):
    '''
    Convert single date/time information to separate columns
    :param data: dataframe to be processed
    :return: dataframe with Dates column expanded to year, month, day, hour, minute
    '''
    data['year'] = data['Dates'].str.slice(2,4).astype(int)
    data['month'] = data['Dates'].str.slice(5,7).astype(int)
    data['day'] = data['Dates'].str.slice(8,10).astype(int)
    data['hour'] = data['Dates'].str.slice(11,13).astype(int)
    data['minute'] = data['Dates'].str.slice(14,16).astype(int)

    return data

def coord_normalization(data):
    '''
    Normalize latitude and longitude values
    :param data: dataframe to be processed
    :return: dataframe with X and Y columns replaced with normalized float values
    '''
    def normalizer(data, col):
        colmean = data[col].mean(axis=0)
        colstd = data[col].std(axis=0)
        data[col] = data[col].apply(lambda x: (x - colmean)/colstd)
        return data

    for col in ['X', 'Y']:
        data = normalizer(data, col)

    return data


def make_street_dict(name):
    '''
    create dictionary of street names with running count
    :param name:
    :param street_count:
    :return: int street_count
    '''
    global street_dict
    global street_count
    if name not in street_dict:
        street_dict[name] = street_count
        street_count += 1


def address_conversion(data):
    '''
    Parse address column into several columns:
    - street intersection vs. block (boolean)
    - street name 1 - convert to number
    - street name 2 - convert to number
    :param data: dataframe to be processed
    :return: dataframe with addtional columns
    '''

    def block_or_int(str):
        if " / " in str:
            out = str.split(" / ")
            make_street_dict(out[0])
            make_street_dict(out[1])
            return -1, street_dict[out[0]], street_dict[out[1]]
        elif "Block of " in str:
            out = str.split(" Block of ")
            make_street_dict(out[1])
            return int(out[0]), street_dict[out[1]], 0
        else:
            return 0, 0, 0


    data['idx'], data['street_1'], data['street_2'] = zip(*data['Address'].map(block_or_int))

    return data


def shape_train(data):
    '''
    Create final train and label data formats.
    Drop unnecessary columns, separate labels,
    convert to numpy arrays.
    :param data: dataframe to be processed
    :return: two shuffled numpy arrays: train and labels
    '''
    data.drop('Dates', axis=1, inplace=True)
    data.drop('Address', axis=1, inplace=True)
    data.drop('Descript', axis=1, inplace=True)
    data.drop('Resolution', axis=1, inplace=True)


    labels = data['Category'].values
    print "label shape", labels.shape
    data.drop('Category', axis=1, inplace=True)
    print "Final Train Columns"
    print data.head(3)
    train_set = data.values
    print "train shape", train_set.shape

    #shuffle after convert to np; use df.values
    np.random.seed(0)
    shuffle = np.random.permutation(np.arange(train_set.shape[0]))
    train_shuf, label_shuf = train_set[shuffle], labels[shuffle]

    return train_shuf, label_shuf

def shape_test(data):
    '''
    Create final test data formats.
    Drop unnecessary columns and convert to numpy array.
    :param data:
    :return: numpy array of test data
    '''
    data.drop('Dates', axis=1, inplace=True)
    data.drop('Address', axis=1, inplace=True)
    
    print "Final Test Columns"
    print data.head(3)
    test_set = data.values
    print "final test shape", test_set.shape

    return test_set


def assemble_train(filename):
    '''
    Umbrella function to do feature engineering on train set.
    :param filename: data source file to read. CSV format.
    :return: dataframe with engineered features.
    '''
    data = df_read(filename)
    data = dayofweek_encode(data) #day of week to int
    data = district_encode(data) #district to int
    data = category_encode(data) #category to int
    data = date_expansion(data) #expand and separate date info
    data = coord_normalization(data) #normalize latitude and longitude
    data = address_conversion(data) #convert address info to int

    return data

def assemble_test(filename):
    '''
    Umbrella function to do feature engineering on test set.
    :param filename: name of test set file to read in.
    :return: df of test data for modeling
    '''
    data = df_read(filename)
    data = dayofweek_encode(data) #day of week to int
    data = district_encode(data)
    data = date_expansion(data) #expand and separate date info
    data = coord_normalization(data) #normalize latitude and longitude
    data = address_conversion(data) #convert address info to int

    return data

if __name__ == "__main__":

    filename = "../data/train.csv"
    testname = "../data/test.csv"

    train_data = assemble_train(filename)

    series1 = train_data.groupby("Category").count(axis=0)["DayOfWeek"]
    print series1.order(ascending = False)
    print train_data.columns.values

    train_set, train_labels = shape_train(train_data)


    for value in train_data.columns.values:
        print value
        print train_data[value].iloc[:10]


