import numpy as np
import pandas
import scipy, scipy.spatial
import sklearn
import sklearn.svm

import sys
import argparse
import pickle

ndim = 900


#r = calClassStat('/home/vahid/Downloads/data/ml/data_train.txt', y[0])

### Calclulate Standardized Mean Difference Between Classes

def calStandMeanDiff(y, cstat, yneg, ypos):
    sx  = np.zeros(shape=ndim, dtype=float)
    ssx = np.zeros(shape=ndim, dtype=float)


    n1 = np.sum(np.in1d(y, yneg))
    n2 = np.sum(np.in1d(y, ypos))
    sys.stderr.write("Number of samples in NegClass: %d and PosClass: %d \n"%(n1, n2))

    for yi in yneg:
        sx += cstat[yi][0]
        ssx += cstat[yi][1]
    r1_mean = sx / float(n1)
    r1_var = (ssx - 2*sx*r1_mean + r1_mean**2) / float(n1)

    sx  = np.zeros(shape=ndim, dtype=float)
    ssx = np.zeros(shape=ndim, dtype=float)
    for yi in ypos:
        sx += cstat[yi][0]
        ssx += cstat[yi][1]
    r2_mean = sx / float(n2)
    r2_var = (ssx - 2*sx*r2_mean + r2_mean**2) / float(n2)

    tot_mean = cstat['all'][0] / float(cstat['all'][2])
    tot_var  = (cstat['all'][1] - 2*cstat['all'][0]*tot_mean + tot_mean**2) / float(cstat['all'][2])

    rdiff = (r1_mean - r2_mean) / np.sqrt(tot_var)

    return (rdiff)


#--------------------------------------#
#                 MAIN                 #
#--------------------------------------#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='Training Data')
    parser.add_argument('labels', help='Training Labels')
    parser.add_argument('test', help='Test Data')
    args = parser.parse_args()

    y = pandas.read_table(args.labels, sep=" ", dtype='int', header=None)
    print(y.head())

    #r = calClassStat(args.train, y[0])
    #print(r)

    cstat = pickle.load( open( "data/sum_features.dat", "rb" ))
    print(cstat[1][0][1:10])
    print(cstat['all'][1][1:10])

    mean_test = calStandMeanDiff(y, cstat, np.arange(1,157), np.arange(157, 165))
    print(np.sum(mean_test > 0.1))


if __name__ == '__main__':
    main()
