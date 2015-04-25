import numpy as np
import pandas
import scipy, scipy.spatial
import sklearn
import sys


from sklearn import linear_model
from sklearn.metrics import precision_score, recall_score, f1_score

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='Training Data')
    parser.add_argument('labels', help='Training Labels')
    parser.add_argument('test', help='Test Data')

    parser.add_argument('data_cv', help='Data for CrossValidation')
    parser.add_argument('label_cv', help='Labels for CrossValidation')

    parser.add_argument('plab', type=int, help='The class to be predicted')
    parser.add_argument('out', help='Output file name')

    args = parser.parse_args()

    y_all = pandas.read_table(args.labels, header=None, sep=' ')
    print(y_all.head())

    ndim = pandas.read_table(args.train, sep=' ', header=None, nrows=3).shape[1]


    featstat = pandas.read_csv('data/feat_stats.csv')
    print(featstat.head())


    # ## Logistic Regression

    clf = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=0.001, \
			fit_intercept=True, intercept_scaling=1, class_weight=None, \
			random_state=None, solver='liblinear', max_iter=10000)


    y = np.empty(shape=y_all.shape[0], dtype=int)

    ic = args.plab

    y[np.where(y_all[0] != ic)[0]] = -1
    y[np.where(y_all[0] == ic)[0]] = 1

    print(y.shape, np.sum(y==1), np.sum(y==-1))

    chunks=100000

    for i in range(1):
       sys.stdout.write('%d '%(i))
       n = 0
       for df in pandas.read_table(args.train, sep=' ', header=None, iterator=True, chunksize=chunks):
           n0, n1 = n*chunks, (n+1)*chunks
           if n1 > y.shape[0]:
               n1 = y.shape[0] - n0
           ysub = y[n0:n1]
           #sys.stdout.write('%d (%d-%d) %d\t'%(n, n0, n1, ysub.shape[0]))
           df = (df - featstat['mean']) / featstat['sigma']
    
           clf.fit(df, ysub)
           n += 1
           break



    ### Reading cross-validation set

    Xcv = pandas.read_table(args.data_cv, sep=' ', header=None)
    print(Xcv.shape)

    ycv = pandas.read_table(args.label_cv, sep=' ', header=None)[0]
    ycv[np.where(ycv==ic)[0]] = -1
    ycv[np.where(ycv==ic)[0]] = 1

    print(Xcv.shape, ycv.shape, np.sum(ycv == 1))

    ypred_cv = clf.predict(Xcv)

    prec = precision_score(ycv, ypred_cv)
    rec  = recall_score(ycv, ypred_cv)
    f1score = f1_score(ycv, ypred_cv)

    print(prec, rec, f1score)
    print(np.sum(ypred_cv == 1), np.sum(ycv == 1))


if __name__ == '__main__':
    main()
