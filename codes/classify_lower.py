import numpy as np
import pandas
import scipy, scipy.spatial
import sklearn
import sys
import argparse


def get_label(arr):
    return(y.iloc[arr,1].values)

sys.setrecursionlimit(1500)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_train', help='Datafile')
    parser.add_argument('label_train', help='The label file for training data')
    parser.add_argument('data_cv', help='Datafile for CrossValidation')
    parser.add_argument('label_cv', help='Labels for CrossValidation')
    parser.add_argument('data_test', help='Test dataset')
    parser.add_argument('out', help='Output file')
    parser.add_argument('jobid', help='Job ID number')
    args = parser.parse_args()

    ndim= 900

    global y
    y = pandas.read_table(args.label_train, sep=' ', header=None, dtype='int')

    #ycv = pandas.read_table(args.label_cv, sep=' ', header=None, dtype='int')

    print(np.unique(y[1]))
    #print(np.unique(ycv[1]))

    ntot_train = y.shape[0]

    feat_idx = np.random.choice(ndim, size=30, replace=False)
    sel_rows = np.random.choice(y.shape[0], int(0.2 * ntot_train))

    df = pandas.read_table(args.data_train, usecols=feat_idx, header=None, sep=' ')
    df = df.iloc[sel_rows, :]
    print(df.shape)
    
    Xcv = pandas.read_table(args.data_cv, usecols=feat_idx, header=None, sep=' ')
    print('\n %s %d %d ==> ' %(args.jobid, df.shape[0], Xcv.shape[0]))

    print('%6d-%6d  '%(df.shape[0], df.shape[1]))
    kdt = scipy.spatial.KDTree(df, leafsize=1000)
    print('KDTree is built succesfully!!')

    qt_dist, qt_idx = kdt.query(Xcv, k=10)
    print('Query of XC data finished!!')
    pred_cv = np.apply_along_axis(get_label, 0, qt_idx)

    np.savetxt('%s.%s.dist_cv'%(args.out, args.jobid), qt_dist, fmt='%.4f')
    np.savetxt('%s.%s.pred_cv'%(args.out, args.jobid), pred_cv, fmt='%d')

    Xts = pandas.read_table(args.data_test, usecols=feat_idx, header=None, sep=' ')
    qt_dist, qt_idx = kdt.query(Xts, k=10)

    pred = np.apply_along_axis(get_label, 0, qt_idx)
    np.savetxt('%s.%s.dist'%(args.out, args.jobid), qt_dist, fmt='%.4f')
    np.savetxt('%s.%s.pred'%(args.out, args.jobid), pred, fmt='%d')


if __name__ == '__main__':
    main()
