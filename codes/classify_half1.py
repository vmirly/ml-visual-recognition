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

##########################################
def readRandomSample(data_fname, y, size, goodfeat=None, acc_miny=None, acc_maxy=None):
    """ Read a random sample
    """
    if goodfeat is None:
        goodfeat = np.arange(ndim)
    Xsub = np.empty(shape=(size,goodfeat.shape[0]), dtype=float)
    ysub = np.zeros(shape=size, dtype=int)

    if acc_miny is None:
        acc_miny = np.min(y)
    if acc_maxy is None:
        acc_maxy = np.max(y)

    #yuniq, ycount = np.unique(y, return_counts=True)
    #tot_acceptable = np.sum(ycount[np.where((yuniq >= acc_miny) & (yuniq <= acc_maxy))[0]])

    acceptable_indx = np.where((y>=acc_miny) & (y<=acc_maxy))[0]
    assert(acceptable_indx.shape[0] > size)
    choice_indx = np.sort(np.random.choice(acceptable_indx, size, replace=False))
    print(choice_indx.shape)
    #sys.stderr.write("Total Accetables: --> %d"%(tot_acceptable))

    #proba = 1.0 - size/float(tot_acceptable)


    with open(data_fname, 'r') as fp:
        n = 0
        nf = 0
        for line in fp:
#            if (y[n] >= acc_miny and y[n]<=acc_maxy):
#                if np.random.uniform(low=0, high=1) > proba and nf < size:
            if nf < size:
                if n == choice_indx[nf]:
                    line = line.strip().split()
                    ix = -1
                    for i,v in enumerate(line):
                        if np.any(goodfeat == i):
                            ix += 1
                            Xsub[nf,ix] = int(v)
                    ysub[nf] = y[n]

                    nf += 1
            n += 1
    return(Xsub, ysub)

### Performance Evaluation
def evalPerformance(ytrue, ypred):
    tp = np.sum(ypred[np.where(ytrue ==  1)[0]] == 1)
    fp = np.sum(ypred[np.where(ytrue == -1)[0]] == 1)
    tn = np.sum(ypred[np.where(ytrue == -1)[0]] == -1)
    fn = ytrue.shape[0]-(tp+fp+tn)
    #sys.stderr.write (" (%d %d %d %d)"%(tp, fp, tn, fn))

    prec = tp / float(tp + fp)
    recall  = tp / float(tp + fn)
    f1score = 2*tp / float(2*tp + fp + fn)

    return (prec, recall, f1score)

#--------------------------------------#
#                 MAIN                 #
#--------------------------------------#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='Training Data')
    parser.add_argument('labels', help='Training Labels')
    parser.add_argument('test', help='Test Data')
    parser.add_argument('out', help='Output file name')

    parser.add_argument('data_cv', help='Data for CrossValidation')
    parser.add_argument('label_cv', help='Labels for CrossValidation')

    args = parser.parse_args()

    y = pandas.read_table(args.labels, sep=" ", dtype='int', header=None)
    print(y.head())

    #r = calClassStat(args.train, y[0])
    #print(r)

    cstat = pickle.load( open( "data/sum_features.dat", "rb" ))
    print(cstat[1][0][1:10])
    print(cstat['all'][1][1:10])


    rdiff = calStandMeanDiff(y, cstat, np.arange(1,157), np.arange(157, 165))
    ## Good Features:
    goodfeatures = np.where(rdiff > 0.1)[0]
    print(goodfeatures)

    #gf_test = np.arange(21,35)
    #Xsub, ysub = readRandomSample(args.train, y[0], \
    #                          size=2000, goodfeat=gf_test, acc_miny=15, acc_maxy=20)
    #print(Xsub.shape)
    #print(np.unique(ysub))

    n = 20000
    for i in range(1):
        Xsub, ysub = readRandomSample(args.train, y[0], size=n, goodfeat=goodfeatures)

        ysub[np.where(ysub <= 156)[0]] = -1
        ysub[np.where(ysub  > 156)[0]] =  1

        x_mean = np.mean(Xsub, axis=0)
        x_std = np.std(Xsub, axis=0)
        Xsub = (Xsub - x_mean) / x_std

        sys.stderr.write('Applying SVM classification ... %d'%(i))

        clf = sklearn.svm.SVC(C=1.0, kernel='rbf', gamma=0.0010)
        clf.fit(Xsub, ysub)

        Xcv = pandas.read_table(args.data_cv, sep=' ', usecols=goodfeatures, dtype='int', header=None)
        ytrue_cv = pandas.read_table(args.label_cv, sep=' ', dtype='int', header=None)
        ytrue_cv[np.where(ytrue_cv <= 156)[0]] = -1
        ytrue_cv[np.where(ytrue_cv  > 156)[0]] =  1
        Xcv = (Xcv - x_mean) / x_std
        ypred_cv = clf.predict(Xcv)
        prec, recall, f1score = evalPerformance(ytrue_cv, ypred_cv)
        print('CrossVal-Perf: Prec=%.3f  Recall=%.3f   F1-score=%.3f\n'%(prec, recall, f1score))

        Xtest = pandas.read_table(args.test, sep=" ", usecols=goodfeatures, dtype='int', header=None)
        Xtest = (Xtest - x_mean) / x_std
        sys.stderr.write('Test data  shape=(%d,%d)'%(Xtest.shape[0], Xtest.shape[1]))

        #ypred = np.zeros(shape=Xtest.shape[0], dtype=int)
        ypred = clf.predict(Xtest)
        np.save_txt(args.out, ypred, header='# CrossVal-Perf.: Prec=%.3f_Recall=%.3f_F1-score=%.3f \n'%(prec, recall, f1score))



if __name__ == '__main__':
    main()
