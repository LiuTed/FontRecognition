import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn import svm
from sklearn.cross_validation import train_test_split
import pickle
from skimage.feature import daisy as skdaisy
from sklearn.decomposition import IncrementalPCA

key2idx = {"font": 0, "fontVariant": 1, "m_label": 2, "strength": 3, "italic": 4, "m_top": 5, "m_left": 6, "h": 7,
           "w": 8, "img": 9}


def readCSVAndSerialize(fontList, csvDir='fonts', storeDir='rawData', filt=None, origin=True):
    if fontList is None:
        return
    if not os.path.exists(storeDir):
        os.mkdir(storeDir)

    for i in fontList:
        basename, extname = os.path.splitext(i)
        if extname.lower() != '.csv':
            continue
        imgs = []
        other_info = []
        path = csvDir + '/' + i

        with open(file=path, mode='r') as f:
            reader = csv.DictReader(f)
            for line in reader:
                if filt:  # where we can use multiple filtering
                    for fil in filt:
                        if not fil(line):
                            continue

                h = int(line["h"])
                w = int(line["w"])
                oh = int(line["originalH"])
                ow = int(line["originalW"])
                st = int(line["m_top"])
                sl = int(line["m_left"])

                img = np.empty([h, w], dtype=np.uint8)
                for i in range(h):
                    for j in range(w):
                        img[i, j] = int(line["r%dc%d" % (i, j)])

                if origin:
                    im = Image.fromarray(img)
                    if ow > 64 or oh > 64:
                        maxi = max(ow, oh)
                        neww = int(ow * 64 / maxi)
                        newh = int(oh * 64 / maxi)
                    else:
                        neww = ow
                        newh = oh
                    im_resize = np.array(im.resize([neww, newh]))
                    img = np.zeros([64, 64], dtype=np.uint8)
                    st_tmp = (64 - newh) // 2
                    sl_tmp = (64 - neww) // 2
                    img[st_tmp:st_tmp + newh, sl_tmp:sl_tmp + neww] = im_resize

                # show_img(img)
                other_info.append([line["font"], line["fontVariant"], int(line["m_label"]), float(line["strength"]),
                                   int(line["italic"]), st, sl, h, w])
                imgs.append(img)
        imgs = np.array(imgs)
        np.save(os.path.join(storeDir, basename + '.npy'), imgs)
        with open(os.path.join(storeDir, basename + '.pk'), "wb") as f:
            pickle.dump(other_info, f)


def deserialize(fontList, storeDir='rawData'):
    if fontList is None:
        return

    imgs = []
    other_info = []
    for i in fontList:
        npfile = os.path.join(storeDir, i + '.npy')
        pkfile = os.path.join(storeDir, i + '.pk')
        if not (os.path.exists(npfile) and os.path.exists(pkfile)):
            print('Warning: file %s or %s not exist' % (npfile, pkfile))
            continue
        imgs.append(np.load(npfile))
        with open(pkfile, "rb") as f:
            other_info.append(pickle.load(f))
    return imgs, other_info

def daisy(img):
    return skdaisy(img, radius=8, histograms=4, orientations=4)

def svmClassifier(X, y, fonts, cv_times=10, test_size=0.25, kernel='rbf', C=1.0, show=True):
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
                accuracy[times][j][0] = clf.score(X=np.array(X_train)[s], y=np.array(y_train)[s])
                if show:
                    print('Train accuracy for "' + fonts[j] + '" is: ', accuracy[times][j][0])
            s = []
            for i in range(len(y_test)):
                if y_test[i] == fonts[j]:
                    s.append(i)
            if s:
                accuracy[times][j][1] = clf.score(X=np.array(X_test)[s], y=np.array(y_test)[s])
                if show:
                    print('Test accuracy for "' + fonts[j] + '" is: ', accuracy[times][j][1])
    return accuracy

def main():
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
    # readCSV(origin = True)

    fonts = ['CALIFORNIAN', 'HARRINGTON', 'BRUSH', 'MODERN', 'PAPYRUS', 'EDWARDIAN', 'FREESTYLE']
    # fonts = ['BRUSH', 'PAPYRUS']

    #'''-----SELECT FROM RAWDATA-----
    paths = [f + '.csv' for f in fonts]
    filt = (lambda x: int(x['m_label']) < 128,
            lambda x: x['fontVariant'] != 'scanned',
            lambda x: x['orientation'] == 0)
    readCSVAndSerialize(fontList=paths, filt=filt)
    #'''

    imgs, other_info = deserialize(fontList=fonts)

    M = len(imgs)
    y = []
    X = []
    for i in range(M):
        N = len(imgs[i])
        for j in range(N):
            y.append(fonts[i])
            d = daisy(imgs[i][j])
            X.append(list(d.reshape(np.size(d))))
            print(str(i) + '/' + str(j))

    n_components = 100
    clf = IncrementalPCA(n_components=n_components, whiten=True)
    X = clf.fit_transform(X=X)

    acc = svmClassifier(X, y, fonts, show=False)
    print(acc)

if __name__ == '__main__':
    main()