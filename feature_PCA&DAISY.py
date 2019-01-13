import utils
import numpy as np
from sklearn.decomposition import IncrementalPCA
from SVM import svmClassifier
from NB import ensembleNB
import matplotlib.pyplot as plt

def main():
    fonts = ['CALIFORNIAN', 'HARRINGTON', 'BRUSH', 'MODERN', 'PAPYRUS', 'EDWARDIAN', 'FREESTYLE']

    '''-----PCA FEATURE-----
    imgs, other_info = utils.deserialize('rawData', fonts)
    X = []
    y = []
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            y.append(fonts[i])
            X.append(list(np.reshape(imgs[i][j], np.size(imgs[i][j]))))
    y = np.array(y)
    X = np.array(X)

    pca = IncrementalPCA(n_components=200, whiten=True)
    X_pca = pca.fit_transform(X=X)
    np.save('proData/base7_y.npy', y)
    np.save('proData/base7_200pca_X.npy', X_pca)
    '''

    '''-----PLOTING PCA-----
    y = np.load('proData/base7_y.npy')
    X_pca = np.load('proData/base7_200pca_X.npy')
    X_20_21 = {}
    for i in range(len(y)):
        if y[i] not in X_20_21.keys():
            X_20_21[y[i]] = [[X_pca[i][19], X_pca[i][20]]]
        else:
            X_20_21[y[i]].append([X_pca[i][19], X_pca[i][20]])
    for font in fonts:
        X_20_21[font] = np.array(X_20_21[font]).transpose()
        plt.scatter(X_20_21[font][0], X_20_21[font][1], label=font, alpha=0.3, marker='o', s=10)
    plt.title('The 20 & 21st PCA features.')
    plt.legend(loc='lower right')
    plt.savefig('pca_20_21_7sorts.png')
    plt.close()
    '''

    '''-----PLOTING DAISY-----
    y = np.load('proData/daisy7_y.npy')
    X_pca = np.load('proData/daisy7_100pca_X.npy')
    X_20_21 = {}
    for i in range(len(y)):
        if y[i] not in X_20_21.keys():
            X_20_21[y[i]] = [[X_pca[i][19], X_pca[i][20]]]
        else:
            X_20_21[y[i]].append([X_pca[i][19], X_pca[i][20]])
    for font in fonts:
        X_20_21[font] = np.array(X_20_21[font]).transpose()
        plt.scatter(X_20_21[font][0], X_20_21[font][1], label=font, alpha=0.3, marker='o', s=10)
    plt.title('The 20 & 21st DAISY features.')
    plt.legend(loc='lower right')
    plt.savefig('daisy_20_21_7sorts.png')
    plt.close()
    '''

    '''-----0 RATIO-----
    imgs, other_info = utils.deserialize('rawData', ['BRUSH'])
    ratio = 0
    for img in imgs[0]:
        count = 0
        for p in np.reshape(img, np.size(img)):
            if p > 0.1:
                count += 1
        ratio += count / np.size(img)
    ratio /= len(imgs[0])
    print(ratio)

    X = np.load('proData/daisy7_X.npy')
    ratio = 0
    m = np.mean(X)
    for img in X:
        count = 0
        for p in img:
            if p > m:
                count += 1
        ratio += count / len(img)
    print(ratio / len(X))
    '''

    '''-----PCA VS DAISY-----
    y = np.load('proData/base7_y.npy')
    X = np.load('proData/base7_200pca_X.npy')
    n_components = 50
    acc1 = []
    Cs = [20] * 7
    for C in Cs:
        print('C =', C)
        ai = svmClassifier(X=X[..., :n_components], y=y, fonts=fonts, C=C, show=False, average=True)
        acc1.append(list(ai.transpose()[1]))
        print(acc1[-1])
    acc1 = np.array(acc1).transpose()

    y = np.load('proData/daisy7_y.npy')
    X = np.load('proData/daisy7_100pca_X.npy')
    acc2 = []
    Cs = [20] * 7
    for C in Cs:
        print('C =', C)
        ai = svmClassifier(X=X[..., :n_components], y=y, fonts=fonts, C=C, show=False, average=True)
        acc2.append(list(ai.transpose()[1]))
        print(acc2[-1])
    acc2 = np.array(acc2).transpose()
    
    plt.subplot(121)
    for i in range(len(fonts)):
        plt.plot(list(range(1, 8)), acc1[i], label=fonts[i])
    plt.ylim((0.65, 1.05))
    plt.title('Accuracy of PCA feature\nrbf kernel SVM, n_components=50, C=20')
    plt.ylabel('Accuracy')
    plt.subplot(122)
    for i in range(len(fonts)):
        plt.plot(list(range(1, 8)), acc2[i], label=fonts[i])
    plt.ylim((0.65, 1.05))
    plt.title('Accuracy of DAISY feature\nrbf kernel SVM, n_components=50, C=20')
    plt.legend(loc='lower right')
    plt.savefig('svm_pcaBase.png')
    plt.close()
    #'''

if __name__ == '__main__':
    main()