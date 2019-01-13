import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
from SVM import textScore, daisyDeserialize
import random

# nbClassifier(): This function presents the SVM method we apply here where 'X' and 'y' is the training data.
# 'fonts' is the list of font sorts included in 'y'. 'cv_times' is the times of fold of cross-validation.
# 'test_size' is the ratio by which we split the data for cross-validation.
# 'nb_clf' is the NB classifier we use and is set with Gaussian NB as default.
# 'show' represents if the function provides a progress report.
# 'test_num' is the length of sentence we intend to consider to generate the accuracy.
# 'average' represents whether the function return the average accuracy of each sort.
def nbClassifier(X, y, fonts, nb_clf=GaussianNB, cv_times=10, test_size=0.25, show=False, text_num=1, average=False):
    accuracy = np.zeros((10, len(fonts), 2))  # to record the outcome
    for times in range(cv_times):
        if show:
            print('\n\n' + str(times + 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        clf = nb_clf()
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

# ensembleNB(): This is the function realizing ensemble strategy. 'X' and 'y' are the training data.
# 'nb_clf' is the NB classifier we use. 'nb_num' means the number of weak classifier to generate.
# 'feature_ratio' is the ratio by which we randomly select DAISY features and construct weak classifier.
# 'n_pca' is the number of PCA components that we extract from selected DAISY features.
# 'show' represents if the function provides a progress report.
def ensembleNB(X, y, nb_clf=GaussianNB, nb_num=10, feature_ratio=0.1, n_pca=10, show=False):
    clf = []
    M = len(X[0])
    m = int(M * feature_ratio)
    for i in range(nb_num):
        clf_i = {'s': sorted(random.sample(range(M), m))}
        X1 = X[..., clf_i['s']]
        clf_i['pca'] = IncrementalPCA(n_components=n_pca, whiten=True).fit(X=X1)
        X_pca = clf_i['pca'].transform(X1)
        clf_i['clf'] = nb_clf().fit(X=X_pca, y=y)
        clf.append(clf_i)
        if show is True:
            print('nb classifier ' + str(i) + ' constructed...')
    return clf

# refinedEnsembleNB(): This function is the refined version of ensembleNB(), however, fails to make much improvement.
# 'P'/'Q'/'R' are width/length/height of the three-order DAISY descriptor.
# The refined strategy is that instead of selecting from the entire descriptor, we only select the third
# dimension of the tensor.
def refinedEnsembleNB(X, y, P=12, Q=12, R=52, nb_clf=GaussianNB, nb_num=10, feature_ratio=0.1, n_pca=10, show=False):
    clf = []
    N = len(X)
    m = int(R * feature_ratio)
    X_raw = np.reshape(X, (N, P, Q, R))
    for i in range(nb_num):
        clf_i = {'s': sorted(random.sample(range(R), m))}
        X1 = X_raw[..., clf_i['s']].copy()
        X1 = np.reshape(X1, (N, P * Q * len(clf_i['s'])))
        clf_i['pca'] = IncrementalPCA(n_components=n_pca, whiten=True).fit(X=X1)
        X_pca = clf_i['pca'].transform(X1)
        clf_i['clf'] = nb_clf().fit(X=X_pca, y=y)
        clf.append(clf_i)
        if show is True:
            print('nb classifier ' + str(i) + ' constructed...')
    return clf

# ensembleScore(): This is the function for scoring ensemble svm. 'clf_seq' represents the sequence of weak classifiers.
# 'X' and 'y' are the test data. 'fonts'is the list of font sort included in 'y'.
def ensembleScore(clf_seq, X, y, fonts):
    N = len(y)
    n = len(fonts)
    y_ = []
    for clf_i in clf_seq:
        X_pca = clf_i['pca'].transform(X[..., clf_i['s']])
        y_.append(list(clf_i['clf'].predict(X_pca)))
    y_ = np.array(y_).transpose()
    correct = [0] * n
    total = [0] * n
    for i in range(N):
        mark = 0
        for k in range(n):
            if y[i] == fonts[k]:
                mark = k
                break
        total[mark] += 1
        yhat = sorted([(np.sum(np.array(y_[i]) == f), f) for f in set(y_[i])])[-1][1]
        if yhat == y[i]:
            correct[mark] += 1
    for k in range(n):
        correct[k] /= total[k]
    return correct

# refinedEnsembleScore(): This function is the scoring function for refinedEnsembleNB() similar to ensembleScore().
def refinedEnsembleScore(clf_seq, X, y, fonts, P=12, Q=12, R=52):
    N = len(y)
    n = len(fonts)
    y_ = []
    X_raw = np.reshape(X, (N, P, Q, R))
    for clf_i in clf_seq:
        X1 = X_raw[..., clf_i['s']]
        X_pca = clf_i['pca'].transform(np.reshape(X1, (N, P * Q * len(clf_i['s']))))
        y_.append(list(clf_i['clf'].predict(X_pca)))
    y_ = np.array(y_).transpose()
    correct = [0] * n
    total = [0] * n
    for i in range(N):
        mark = 0
        for k in range(n):
            if y[i] == fonts[k]:
                mark = k
                break
        total[mark] += 1
        yhat = sorted([(np.sum(np.array(y_[i]) == f), f) for f in set(y_[i])])[-1][1]
        if yhat == y[i]:
            correct[mark] += 1
    for k in range(n):
        correct[k] /= total[k]
    return correct

def twoSortComparison(font1, font2, X, y, nb_num=10, rep=7, feature_ratio=0.05, n_pca=30):
    acc = []
    for i in range(rep):
        clf = ensembleNB(X=X, y=y, nb_num=nb_num, feature_ratio=feature_ratio, n_pca=n_pca)
        acc.append(ensembleScore(clf_seq=clf, X=X, y=y, fonts=[font1, font2]))
    acc = np.array(acc).transpose()
    plt.plot(list(range(1, rep + 1)), acc[0], label=font1)
    plt.plot(list(range(1, rep + 1)), acc[1], label=font2)
    plt.title('Acc of two sorts: ' + font1 + ' & ' + font2 + '\nnb_num=' + str(nb_num) + ', feature_ratio=' + str(feature_ratio) + ', n_pca=' + str(n_pca))
    plt.legend(loc='lower right')
    plt.savefig('nb_twoSort_' + font1 + '_' + font2 + '.png')
    plt.close()

def main():
    fonts = ['CALIFORNIAN', 'HARRINGTON', 'BRUSH', 'MODERN', 'PAPYRUS', 'EDWARDIAN', 'FREESTYLE']
    # fonts = ['BRUSH', 'FREESTYLE', 'VLADIMIR']
    # fonts = ['CALIFORNIAN', 'HARRINGTON', 'BRUSH', 'MODERN', 'PAPYRUS', 'EDWARDIAN', 'FREESTYLE', 'AGENCY', 'BASKERVILLE', 'BAUHAUS', 'VLADIMIR', 'VIVALDI', 'VINER', 'TXT']

    '''-----BASIC NB-----
    X_pca = np.load(file='proData/daisy14_200pca_X.npy')
    acc = nbClassifier(X=X_pca[..., :100], y=y, fonts=fonts, average=True)
    print(acc)
    '''

    '''-----SIMPLE ENSEMBLE-----
    X_ = []
    y_ = []
    for i in range(len(y)):
        if y[i] in fonts:
            y_.append(y[i])
            X_.append(list(X[i]))
    y = np.array(y_)
    X = np.array(X_)
    '''
    '''
    # nums = [10, 20, 30, 40, 50, 60, 70]
    # nums = [1, 2, 3, 4, 5, 6, 7]
    y, X = daisyDeserialize(filename='daisy14')
    nums = [20] * 7
    acc1 = []
    for num in nums:
        print('num =', num)
        clf = ensembleNB(X=X, y=y, nb_num=num, feature_ratio=0.05, n_pca=30)
        acc1.append(ensembleScore(clf_seq=clf, X=X, y=y, fonts=fonts))
    acc1 = np.array(acc1).transpose()

    X_pca = np.load(file='proData/daisy14_200pca_X.npy')
    acc2 = []
    for i in range(7):
        print('times =', i)
        ai = nbClassifier(X=X_pca[..., :30], y=y, fonts=fonts, average=True)
        acc2.append(list(ai.transpose()[1]))
    acc2 = np.array(acc2).transpose()

    plt.subplot(121)
    for i in range(len(fonts)):
        plt.plot(list(range(1, 8)), acc2[i], label=fonts[i])
    plt.ylim((0.05, 0.65))
    plt.title('Accuracy of NB\nnb, n_components=30')
    plt.ylabel('Accuracy')
    plt.subplot(122)
    for i in range(len(fonts)):
        plt.plot(list(range(1, 8)), acc1[i], label=fonts[i])
    plt.ylim((0.05, 0.65))
    plt.title('Accuracy of ensemble NB\nnb, feature_ratio=0.05\nnb_num=20, n_components=30')
    plt.legend(loc='lower right')
    plt.savefig('nb_normalVSensemble.png')
    plt.close()
    '''

    '''-----REFINED ENSEMBLE-----
    # nums = [10, 20, 30, 40, 50, 60, 70]
    y, X = daisyDeserialize(filename='daisy14')
    nums = [50] * 7
    acc1 = []
    for num in nums:
        print('num =', num)
        clf = ensembleNB(X=X, y=y, nb_num=num, feature_ratio=0.05, n_pca=50)
        acc1.append(ensembleScore(clf_seq=clf, X=X, y=y, fonts=fonts))
    acc1 = np.array(acc1).transpose()

    acc2 = []
    for num in nums:
        print('num =', num)
        clf = refinedEnsembleNB(X=X, y=y, nb_num=num, feature_ratio=0.05, n_pca=50)
        acc2.append(refinedEnsembleScore(clf_seq=clf, X=X, y=y, fonts=fonts))
    acc2 = np.array(acc2).transpose()

    plt.subplot(121)
    for i in range(len(fonts)):
        plt.plot(list(range(1, 8)), acc1[i], label=fonts[i])
    plt.ylim((0.20, 0.70))
    plt.title('Accuracy of ensemble NB\nnb, feature_ratio=0.05\nnb_num=50, n_components=50')
    plt.ylabel('Accuracy')
    plt.subplot(122)
    for i in range(len(fonts)):
        plt.plot(list(range(1, 8)), acc2[i], label=fonts[i])
    plt.ylim((0.20, 0.70))
    plt.title('Accuracy of refined ensemble NB\nnb, feature_ratio=0.05\nnb_num=50, n_components=50')
    plt.legend(loc='lower right')
    plt.savefig('nb_ensembleVSrefined.png')
    plt.close()
    '''

if __name__ == '__main__':
    main()