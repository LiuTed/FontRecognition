import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from skimage.feature import daisy
from sklearn.decomposition import IncrementalPCA
import utils

# textScore(): This function is used to give the accuracy of a SVM classifier corresponding to parameter 'clf'.
# 'X' is the test data containing x, and 'font' provides the correct font sort, y.
# 'test_num' represents the length of sentence to test.
# 'rep' is the times of repetition of randomly selecting samples from 'X'.
def textScore(clf, X, font, text_num=1, rep=None):
    n = len(X)
    if rep is None:
        rep = n
    correct = 0
    for i in range(rep):
        s = random.sample(range(n), text_num)
        X0 = X[s]
        y1 = clf.predict(X=X0)
        y_ = sorted([(np.sum(np.array(y1) == f), f) for f in set(y1)])[-1][1]
        if y_ == font:
            correct += 1
    return correct/rep

# svmClassifier(): This function presents the SVM method we apply here where 'X' and 'y' is the training data.
# 'fonts' is the list of font sorts included in 'y'. 'cv_times' is the times of fold of cross-validation.
# 'test_size' is the ratio by which we split the data for cross-validation.
# 'kernel'/'C' are parameters of SVM. 'show' represents if the function provides a progress report.
# 'test_num' is the length of sentence we intend to consider to generate the accuracy.
# 'average' represents whether the function return the average accuracy of each sort.
def svmClassifier(X, y, fonts, cv_times=10, test_size=0.25, kernel='rbf', C=1.0, show=False, text_num=1, average=False):
    accuracy = np.zeros((10, len(fonts), 2))  # to record the outcome
    for times in range(cv_times):
        if show:
            print('\n\n' + str(times + 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        clf = svm.SVC(kernel=kernel, C=C, decision_function_shape='ovo')
        clf.fit(X=X_train, y=y_train)
        for j in range(len(fonts)):
            s = []
            for i in range(len(y_train)):
                if y_train[i] == fonts[j]:
                    s.append(i)
            if s:
                accuracy[times][j][0] = textScore(clf, X_train[s], fonts[j], text_num=text_num)
                if show:
                    print('Train accuracy for "' + fonts[j] + '" is: ', accuracy[times][j][0])
            s = []
            for i in range(len(y_test)):
                if y_test[i] == fonts[j]:
                    s.append(i)
            if s:
                accuracy[times][j][1] = textScore(clf, X_test[s], fonts[j], text_num=text_num)
                if show:
                    print('Test accuracy for "' + fonts[j] + '" is: ', accuracy[times][j][1])
    if not average:
        return accuracy
    else:
        s = None
        for a in accuracy:
            if s is not None:
                s += a
            else:
                s = np.array(a)
        return s/cv_times

def daisyAndSerialize(imgs, fonts, storeDir='proData', filename='daisy', radius=8, histograms=4, orientations=4):
    if not os.path.exists(storeDir):
        os.mkdir(storeDir)
    M = len(imgs)
    y = []
    X = []
    for i in range(M):
        N = len(imgs[i])
        for j in range(N):
            y.append(fonts[i])
            d = daisy(imgs[i][j], radius=radius, histograms=histograms, orientations=orientations)
            X.append(list(d.reshape(np.size(d))))
            print(str(i) + '/' + str(j))
    np.save(os.path.join(storeDir, filename + '_y.npy'), y)
    np.save(os.path.join(storeDir, filename + '_X.npy'), X)

def daisyDeserialize(storeDir='proData', filename='daisy'):
    return np.load(os.path.join(storeDir, filename + '_y.npy')),\
           np.load(os.path.join(storeDir, filename + '_X.npy'))

def main():
    # fonts = ['CALIFORNIAN', 'HARRINGTON', 'BRUSH', 'MODERN', 'PAPYRUS', 'EDWARDIAN', 'FREESTYLE', 'AGENCY', 'BASKERVILLE', 'BAUHAUS', 'VLADIMIR', 'VIVALDI', 'VINER', 'TXT']
    fonts = ['CALIFORNIAN', 'HARRINGTON', 'BRUSH', 'MODERN', 'PAPYRUS', 'EDWARDIAN', 'FREESTYLE']

    # y = np.load('proData/daisy7_y.npy')
    # X = np.load('proData/daisy7_100pca_X.npy')

    '''-----SELECT FROM RAWDATA-----
    paths = [f + '.csv' for f in fonts]
    filt = None
    utils.readCSVAndSerialize(csvDir='fonts', storeDir='rawData', csvList=paths, filt=filt)
    '''

    '''-----PROCESS BY DAISY-----
    imgs, other_info = utils.deserialize(storeDir='rawData', fontList=fonts)
    daisyAndSerialize(imgs, fonts, filename='daisy7')
    y, X = daisyDeserialize(filename='daisy7')
    pca = IncrementalPCA(n_components=100, whiten=True)
    X_pca = pca.fit_transform(X=X)
    np.save('proData/daisy7_100pca_X.npy', X_pca)
    '''

    '''-----C VARIATION-----
    n_components = 50
    acc = []
    Cs = [1 * i for i in range(1, 8)]
    for C in Cs:
        print('C =', C)
        ai = svmClassifier(X=X[..., :n_components], y=y, fonts=fonts, C=C, show=False, average=True)
        acc.append(list(ai.transpose()[1]))
        print(acc[-1])
    acc = np.array(acc).transpose()
    ax = plt.subplot(111)
    for i in range(len(fonts)):
        ax.plot(Cs, acc[i], label=fonts[i])
    plt.title('Accuracy with respect to C\nrbf kernel SVM, n_components=50')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('svm_CVariation.png')
    plt.close()
    '''

    '''-----N COMPONENTS VARIATION-----
    C = 20
    acc = []
    ns = [10 * i for i in range(1, 8)]
    for n in ns:
        print('n =', n)
        ai = svmClassifier(X=X[..., :n], y=y, fonts=fonts, C=C, show=False, average=True)
        acc.append(list(ai.transpose()[1]))
        print(acc[-1])
    acc = np.array(acc).transpose()
    ax = plt.subplot(111)
    for i in range(len(fonts)):
        ax.plot(ns, acc[i], label=fonts[i])
    plt.title('Accuracy with respect to n_components\nrbf kernel SVM, C=20')
    plt.xlabel('n_components')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('svm_nVariation.png')
    plt.close()
    '''

    '''-----TEXT ACCURACY-----
    nums = [1, 5, 10, 20, 30, 50, 100]
    C = 1.0
    n_pca = 10
    acc = []
    for num in nums:
        print('num =', num)
        ai = svmClassifier(X=X[..., :n_pca], y=y, fonts=fonts, C=C, text_num=num, show=False, average=True)
        acc.append(list(ai.transpose()[1]))
        print(acc[-1])
    acc = np.array(acc).transpose()
    ax = plt.subplot(111)
    xaxis = list(range(len(nums)))
    for i in range(len(fonts)):
        ax.plot(xaxis, acc[i], label=fonts[i])
    plt.xticks(xaxis, nums)
    plt.title('Accuracy with respect to text_num\nrbf kernel SVM, C=1.0, n_components=10')
    plt.xlabel('text_num')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('svm_textVariation.png')
    plt.close()
    '''

if __name__ == '__main__':
    main()