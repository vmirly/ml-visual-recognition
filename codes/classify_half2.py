import numpy as np
import pandas
import scipy, scipy.spatial
import sklearn
import sklearn.svm

import sys
import argparse
import pickle



sys.path.append('codes/')
from utilities import *


#r = calClassStat('/home/vahid/Downloads/data/ml/data_train.txt', y[0])

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


    rdiff = calStandMeanDiff(y, cstat, np.arange(157,162), np.arange(162, 165))
    ## Good Features:
    goodfeatures = np.where(rdiff > 0.1)[0]
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
                goodfeat=goodfeatures, acc_miny=157, acc_maxy=164)
        assert(np.sum(ysub < 157) == 0)
        ysub[np.where(ysub <  162)[0]] = -1
        ysub[np.where(ysub >= 162)[0]] =  1

        x_mean = np.mean(Xsub, axis=0)
        x_std = np.std(Xsub, axis=0)
        Xsub = (Xsub - x_mean) / x_std

        sys.stderr.write('Applying SVM classification ... %d'%(i))

        clf = sklearn.svm.SVC(C=0.8, kernel='rbf', gamma=0.0050)
        clf.fit(Xsub, ysub)

        Xcv = pandas.read_table(args.data_cv, sep=' ', usecols=goodfeatures, dtype='int', header=None)
        ytrue_cv = pandas.read_table(args.label_cv, sep=' ', dtype='int', header=None)[0]
        Xcv = Xcv.iloc[np.where(ytrue_cv >= 157)[0],:]
        print('CrossVal Shape= %d,%d' %Xcv.shape)
        ytrue_cv = ytrue_cv[np.where(ytrue_cv >= 157)[0]].values
        ytrue_cv[np.where(ytrue_cv <  162)[0]] = -1
        ytrue_cv[np.where(ytrue_cv >= 162)[0]] =  1
        Xcv = (Xcv - x_mean) / x_std
        ypred_cv = clf.predict(Xcv)
        prec, recall, f1score = evalPerformance(ytrue_cv, ypred_cv)
        print('CrossVal-Perf: Prec=%.3f  Recall=%.3f   F1-score=%.3f\n'%(prec, recall, f1score))

        np.savetxt('%s.cv'%args.out, ypred_cv, fmt='%d', \
            header=' CrossVal-Perf.: Prec %.3f Recall %.3f F1-score %.3f (n= %d )'%(prec, recall, f1score, n))
        Xtest = pandas.read_table(args.test, sep=' ', usecols=goodfeatures, dtype='int', header=None)
        Xtest = (Xtest - x_mean) / x_std
        sys.stderr.write('Test data  shape=(%d,%d)'%(Xtest.shape[0], Xtest.shape[1]))

        #ypred = np.zeros(shape=Xtest.shape[0], dtype=int)
        ypred = clf.predict(Xtest)
        np.savetxt(args.out, ypred, fmt='%d', \
            header=' CrossVal-Perf.: Prec %.3f Recall %.3f F1-score %.3f (n= %d )'%(prec, recall, f1score, n))



if __name__ == '__main__':
    main()
