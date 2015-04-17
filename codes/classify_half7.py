import numpy as np
import pandas
import scipy, scipy.spatial
import sklearn
import sklearn.svm

import sys
import argparse
import pickle



sys.path.append('codes/')
from utilities_v2 import *


ymin = 157
ysplit = 159
ymax = 159

feat_threshold = 0.001

#C=15.0000 Gamma=0.0015  ==> Prec:0.670  Recall:0.734  F1Score:0.700
optimal_c = 15.00
optimal_gamma = 0.0015

#--------------------------------------#
#                 MAIN                 #
#--------------------------------------#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='Training Data')
    parser.add_argument('labels', help='Training Labels')
    parser.add_argument('test', help='Test Data')

    parser.add_argument('data_cv', help='Data for CrossValidation')
    parser.add_argument('label_cv', help='Labels for CrossValidation')

    parser.add_argument('out', help='Output file name')

    args = parser.parse_args()

    y = pandas.read_table(args.labels, sep=" ", dtype='int', header=None)
    print(y.head())

    #r = calClassStat(args.train, y[0])
    #print(r)

    cstat = pickle.load( open( "data/sum_features.dat", "rb" ))
    print(cstat[1][0][1:10])
    print(cstat['all'][1][1:10])


    rdiff = calStandMeanDiff(y, cstat, np.arange(ymin,ysplit), np.arange(ysplit, ymax+1))
    ## Good Features:
    goodfeatures = np.where(rdiff > feat_threshold)[0]
    print(goodfeatures)
    sys.stderr.write('Number of Features: %d'%goodfeatures.shape[0])
    #gf_test = np.arange(21,35)
    #Xsub, ysub = readRandomSample(args.train, y[0], \
    #                          size=2000, goodfeat=gf_test, acc_miny=15, acc_maxy=20)
    #print(Xsub.shape)
    #print(np.unique(ysub))

    n = 50000
    for i in range(1):
        Xsub, ysub = readRandomSample(args.train, y[0], size=n, \
                goodfeat=goodfeatures, acc_miny=ymin, acc_maxy=ymax)
        assert(np.sum(ysub < ymin) == 0)
        assert(np.sum(ysub > ymax) == 0)
        ysub[np.where(ysub <  ysplit)[0]] = -1
        ysub[np.where(ysub >= ysplit)[0]] =  1

	features_idx = np.where(np.std(Xsub, axis=0)> 0.0001)[0]
	print("Number of Good Features: %d"%features_idx.shape[0])
	Xsub = Xsub[:,features_idx]

        x_mean = np.mean(Xsub, axis=0)
        x_std = np.std(Xsub, axis=0)

        Xsub = (Xsub - x_mean) / x_std

        sys.stderr.write('Applying SVM classification ... %d'%(i))

        clf = sklearn.svm.SVC(C=optimal_c, kernel='rbf', gamma=optimal_gamma)
        clf.fit(Xsub, ysub)

        Xcv = pandas.read_table(args.data_cv, sep=' ', usecols=goodfeatures, dtype='int', header=None)
        ytrue_cv = pandas.read_table(args.label_cv, sep=' ', dtype='int', header=None)[0]

        Xcv = Xcv.iloc[np.where((ytrue_cv >= ymin) & (ytrue_cv <= ymax))[0],:]

        ytrue_cv = ytrue_cv[np.where((ytrue_cv >= ymin) & (ytrue_cv <= ymax))[0]].values
        #ytrue_cv = ytrue_cv[np.where(ytrue_cv <= ymax)[0]]
        #Xcv = Xcv.iloc[np.where(ytrue_cv <= ymax)[0],:]
        print('CrossVal Shape= %d,%d' %Xcv.shape)


        print(np.unique(ytrue_cv))

        ytrue_cv[np.where(ytrue_cv <  ysplit)[0]] = -1
        ytrue_cv[np.where(ytrue_cv >= ysplit)[0]] =  1
        print("CrossVal: Neg %s\tPos %d"%(np.sum(ytrue_cv == -1), np.sum(ytrue_cv == 1)))

        Xcv = (Xcv - x_mean) / x_std
        ypred_cv = clf.predict(Xcv)
        prec, recall, f1score = evalPerformance(ytrue_cv, ypred_cv)
        print('CrossVal-Perf: Prec=%.3f  Recall=%.3f   F1-score=%.3f\n'%(prec, recall, f1score))

        np.savetxt('%s.cv'%args.out, ypred_cv, fmt='%d', \
            header=' CrossVal-Perf.: Prec %.3f Recall %.3f F1-score %.3f (n= %d  dim= %d )' \
		%(prec, recall, f1score, n, features_idx.shape[0]))
        Xtest = pandas.read_table(args.test, sep=' ', usecols=goodfeatures, dtype='int', header=None)
        Xtest = (Xtest - x_mean) / x_std
        sys.stderr.write('Test data  shape=(%d,%d)'%(Xtest.shape[0], Xtest.shape[1]))

        #ypred = np.zeros(shape=Xtest.shape[0], dtype=int)
        ypred = clf.predict(Xtest)
        np.savetxt(args.out, ypred, fmt='%d', \
            header=' CrossVal-Perf.: Prec %.3f Recall %.3f F1-score %.3f (n= %d  dim= %d )' \
		%(prec, recall, f1score, n, features_idx.shape[0]))



if __name__ == '__main__':
    main()
