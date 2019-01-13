import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import daisy as skdaisy
import os
import sys
import io
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


def show_img(img):
    plt.figure()
    plt.imshow(img, cmap = 'Greys', vmin = 0, vmax = 255)
    plt.show()
    plt.close()

key2idx = {
    "font": 0,
    "fontVariant": 1,
    "m_label": 2,
    "strength": 3,
    "italic": 4,
    "m_top": 5,
    "m_left": 6,
    "h": 7, #always 20
    "w": 8, #always 20
    "originalH": 9,
    "originalW": 10
}

def readCSV(csvDir, origin = False, filt = lambda x: int(x["m_label"]) < 256, fontList = None):
    '''
    read files which satisfy the filt in fontList from csvDir
    if origin is True, operation of recovering the compressed image to original image
    and resizing (cropping and padding) to 64*64 will be applied
    the original image has size (m_top + originalH, m_left + originalW)
    and the character starts at (m_top // 2, m_left // 2)
    after resized to 64*64 the character start at ((64 - originalH) // 2, (64 - originalW) // 2)

    return: (imgs, other informations)
        imgs: 1D list of numpy array with shape = (L) where
            L = len(fontList)
        and the numpy array has shape = (N, H, W) and dtype = np.uint8 where
            N = num of characters in each file
            H = height of each image (20 if origin is False else 64)
            W = width of each image (20 if origin is False else 64)
        other informations: 3D list with shape = (L, N, 9), L and N is same as above,
            9 is the number of other infos the dataset provides
            see key2idx for the meaning and use it to get each value
    '''
    imgs = []
    other_info = []
    cnt = 0
    if fontList is None:
        fontList = os.listdir(csvDir)

    for i in fontList:
        print("%d/%d" % (cnt, len(fontList)))
        cnt += 1
        basename, extname = os.path.splitext(i)
        if extname.lower() != '.csv':
            continue
        path = os.path.join(csvDir, i) # use os api to keep portability
        img_tmp = []
        info_tmp = []
        with open(file=path, mode='r') as f:
            reader = csv.DictReader(f)
            for line in reader:
                # print(path, line["font"], line["m_label"], chr(int(line["m_label"])))
                # sys.stdout.flush()
                if not filt(line):
                    continue

                h = int(line["h"])
                w = int(line["w"])
                oh = int(line["originalH"])
                ow = int(line["originalW"])
                st = int(line["m_top"])
                sl = int(line["m_left"])

                img = np.empty([h, w], dtype = np.uint8)
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
                    img = np.zeros([64, 64], dtype = np.uint8)
                    st_tmp = (64 - newh) // 2
                    sl_tmp = (64 - neww) // 2
                    img[st_tmp:st_tmp+newh, sl_tmp:sl_tmp+neww] = im_resize

                # show_img(img)
                info_tmp.append([line["font"], line["fontVariant"], int(line["m_label"]), float(line["strength"]), int(line["italic"]), st, sl, h, w, oh, ow])
                img_tmp.append(img)
        imgs.append(np.array(img_tmp))
        other_info.append(info_tmp)
    return imgs, other_info

def readCSVAndSerialize(csvDir, storeDir, csvList = None, filt = lambda x: int(x["m_label"]) < 256, origin = True):
    if csvList is None:
        csvList = os.listdir(csvDir)
    if not os.path.exists(storeDir):
        os.mkdir(storeDir)

    for i in csvList:
        basename, extname = os.path.splitext(i)
        if extname.lower() != '.csv':
            continue
        imgs = []
        other_info = []
        path = os.path.join(csvDir, i)

        with open(file=path, mode='r') as f:
            reader = csv.DictReader(f)
            for line in reader:
                if not filt(line):
                    continue

                h = int(line["h"])
                w = int(line["w"])
                oh = int(line["originalH"])
                ow = int(line["originalW"])
                st = int(line["m_top"])
                sl = int(line["m_left"])

                img = np.empty([h, w], dtype = np.uint8)
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
                    img = np.zeros([64, 64], dtype = np.uint8)
                    st_tmp = (64 - newh) // 2
                    sl_tmp = (64 - neww) // 2
                    img[st_tmp:st_tmp+newh, sl_tmp:sl_tmp+neww] = im_resize

                # show_img(img)
                other_info.append([line["font"], line["fontVariant"], int(line["m_label"]), float(line["strength"]), int(line["italic"]), st, sl, h, w])
                imgs.append(img)
        imgs = np.array(imgs)
        np.save(os.path.join(storeDir, basename + '.npy'), imgs)
        with open(os.path.join(storeDir, basename + '.pk'), "wb") as f:
            pickle.dump(other_info, f)

def deserialize(storeDir, fontList = None):
    if fontList is None:
        fontList = os.listdir(storeDir)
        fontList = list(map(lambda x: os.path.splitext(x)[0], fontList))
        fontList = np.unique(fontList)

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
    return skdaisy(img, radius = 8, histograms = 4, orientations = 4)

def test():
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
    # readCSV(origin = True)
    imgs, infos = readCSV('C:\\Users\\chen\\Desktop\\fontr/', fontList = ['CALIBRI.csv'], filt = lambda x: int(x["m_label"]) < 128, origin=True)
    print(np.shape(imgs[0]))
    print(infos[0][751])
    show_img(imgs[0][751])
    d = daisy(imgs[0][751])

    readCSVAndSerialize('C:\\Users\\chen\\Desktop\\fontr/', '../data', csvList = ['CALIBRI.csv'])
    imgs, infos = deserialize('../data')
    print(np.shape(imgs[0]), infos[0][751])

if __name__ == '__main__':
    fonts=["CALIFORNIAN.csv","HARRINGTON.csv","BRUSH.csv","MODERN.csv","PAPYRUS.csv","EDWARDIAN.csv","FREESTYLE.csv"]
    #import data and extract feature
    for i in range(1,8):
        #please use your own path here
        newimage,infos = readCSV('C:\\Users\\chen\\Desktop\\fontr/', fontList = [fonts[i-1]], filt = lambda x: int(x["m_label"]) < 128, origin=True)
        if i==1:
            image=newimage[0]
            label=np.zeros(len(newimage[0]))
            label=[x+1 for x in label]
        else:
            newlabel=np.zeros(len(newimage[0]))
            newlabel=[x+i for x in newlabel]
            label=np.hstack((label,newlabel))
            image=np.vstack((image,newimage[0]))
    #reshape 64*64 features into 4096 features
    image=np.reshape(image,(len(image),4096))
    print(np.shape(image))
    #split trainset and testset randomly
    x_train, x_test, y_train, y_test = train_test_split(image, label, test_size = 0.2)
    sc=np.zeros(10)
    mid=0.1
    #iterate -log(mid)/log(10) from 1 to 10
    for i in range(10):
        md=30
        dtc = DecisionTreeClassifier(criterion='gini',max_features="auto",min_impurity_decrease=mid,max_depth=md)
        dtc.fit(x_train, y_train)
        y_predict=dtc.predict(x_test)
        mid=mid/10
        sc[i]=dtc.score(x_test,y_test)
        print('Accuracy of DT:',sc[i])
    #plot the line
    plt.plot(range(1,11),sc)
    plt.xlabel('-log(min_inpurity_decrease)/log(10)')
    plt.ylabel('accuracy')
    plt.show()

