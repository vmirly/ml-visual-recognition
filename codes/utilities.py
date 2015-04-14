import numpy as np
import pandas
import sys


ndim = 900

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
    assert(ytrue.shape == ypred.shape)
    tp = np.sum(ypred[np.where(ytrue ==  1)[0]] == 1)
    fp = np.sum(ypred[np.where(ytrue == -1)[0]] == 1)
    tn = np.sum(ypred[np.where(ytrue == -1)[0]] == -1)
    fn = ytrue.shape[0]-(tp+fp+tn)
    #sys.stderr.write (" (%d %d %d %d)"%(tp, fp, tn, fn))

    prec = tp / float(tp + fp)
    recall  = tp / float(tp + fn)
    f1score = 2*tp / float(2*tp + fp + fn)

    return (prec, recall, f1score)
