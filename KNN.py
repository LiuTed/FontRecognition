import utils
import numpy as np
import random

fontList = ['CALIFORNIAN', 'HARRINGTON', 'BRUSH', 'MODERN', 'PAPYRUS', 'EDWARDIAN', 'FREESTYLE']

def main():
    random.seed(23333)
    # imgs, other_infos = utils.deserialize("../data", fontList)
    # imgs_shuffled = []
    # for i in imgs:
    #     indices = [j for j in range(i.shape[0])]
    #     random.shuffle(indices)
    #     imgs_shuffled.append(i[indices])
    # train = np.array([utils.hog(img) for i in imgs_shuffled for img in i[:len(i) * 4 // 5]])
    # train = np.reshape(train, [train.shape[0], -1])
    # train_label = np.array([i for i, _ in enumerate(imgs) for j in _[:len(_) * 4 // 5]])
    # test = np.array([utils.hog(img) for i in imgs_shuffled for img in i[len(i) * 4 // 5:]])
    # test = np.reshape(test, [test.shape[0], -1])
    # test_label = np.array([i for i, _ in enumerate(imgs) for j in _[len(_) * 4 // 5:]])
    
    imgs, other_infos = utils.readCSV("../fonts", True,
        lambda x: int(x['m_label']) < 128 and int(x['m_label']) not in [83,84,85,115,116,117],
        fontList = fontList)
    train = np.array([utils.daisy(img) for i in imgs for img in i])
    train = np.reshape(train, [train.shape[0], -1])
    train_label = np.array([i for i, _ in enumerate(imgs) for j in _])

    imgs, other_infos = utils.readCSV("../fonts", True,
        lambda x: int(x['m_label']) in [83,84,85,115,116,117],
        fontList = fontList)
    test = np.array([utils.daisy(img) for i in imgs for img in i])
    test = np.reshape(test, [test.shape[0], -1])
    test_label = np.array([i for i, _ in enumerate(imgs) for j in _])

    for k in [1,4,8,16,24,32]:
        print('k =', k, end = ' ')
        tp = 0
        for idx, t in enumerate(test):
            lst, dist = utils.nearest_neighbour(t, train)
            cnt = [0 for j in range(len(fontList))]
            for i in lst[:k]:
                cnt[train_label[i]] += 1
            if max(cnt) == cnt[test_label[idx]]:
                tp += 1
        print('acc =', tp / test.shape[0])
        

if __name__ == "__main__":
    main()

