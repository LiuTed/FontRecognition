import utils
import numpy as np
import random

fontList = ['CALIFORNIAN', 'HARRINGTON', 'BRUSH', 'MODERN', 'PAPYRUS', 'EDWARDIAN', 'FREESTYLE']

class UnionFind(object):
    def __init__(self, n):
        self._label = [i for i in range(n)]
        self.ntree = n
        self.classidx = [i for i in range(n)]
    
    def find(self, x):
        if self._label[x] == x:
            return x
        else:
            self._label[x] = self.find(self._label[x])
            return self._label[x]
    
    def union(self, x, y):
        xx = self.find(x)
        yy = self.find(y)
        if xx != yy:
            self._label[yy] = xx
            self.ntree -= 1

    def shrink(self):
        m = {}
        for i in range(len(self._label)):
            r = self.find(i)
            if m.get(r) is not None:
                self.classidx[i] = m[r]
            else:
                self.classidx[i] = len(m)
                m[r] = self.classidx[i]

def euclidean_metric(diff):
    return np.sqrt(np.sum(np.square(diff), axis = -1))

def gaussian_weight(diff, bandwidth, invSigma = None):
    if invSigma is None:
        dist = np.sum(np.square(diff), axis = -1)
    else:
        dist = diff @ invSigma @ (diff.swapaxes(-1, -2))
    return np.where(dist > bandwidth, np.zeros_like(dist), np.exp(-.5 * dist / bandwidth))

def uniform_weight(diff, bandwidth):
    dist = euclidean_metric(diff)
    return np.where(dist > bandwidth, np.zeros_like(dist), np.ones_like(dist))

def meanshift(x, r, maxiter = 10, threshold = 1e-2, weight_func = gaussian_weight, batch_size = 32):
    # x.shape = (n, d)
    x = np.expand_dims(x, 0) # x.shape = (1, n, d) for broadcast
    num = x.shape[1]
    dim = x.shape[2]
    nx = np.empty_like(x) # new x
    iter = 0
    while iter < maxiter or maxiter == 0:
        i = 0
        while i < x.shape[1]:
            print("%d/%d"%(i, x.shape[1]))
            tmp = min(batch_size, x.shape[1] - i) # tmp = real batch size
            b = x[0, i: i + tmp] # get one batch b.shape = (tmp, d)
            b = np.expand_dims(b, 1) # b.shape = (tmp, 1, d)
            d = b - x # d.shape = (tmp, n, d)
            w = weight_func(d, r) # weight w.shape = (tmp, n)
            ws = np.sum(w, 1) # weight sum ws.shape = (tmp)
            ws = np.reshape(ws, [1, -1, 1]) # ws.shape = (1, tmp, 1) for broadcast
            nb = np.matmul(w, x) # new batch nb.shape = (1, tmp, d)
            nb /= ws # divide weight sum
            nx[0, i: i + tmp] = nb
            i += tmp
        
        iter += 1
        print(iter, "done")
        maxdis = np.max(euclidean_metric(nx - x))
        # max shift distance
        x, nx = np.frombuffer(nx), np.frombuffer(x)
        #swap values
        x = np.reshape(x, [1, num, dim])
        nx = np.reshape(nx, [1, num, dim])
        if(maxdis < threshold):
            break # stop
    
    label = UnionFind(num)
    # label each point belongs to which cluster
    i = 0
    while i < x.shape[1]:
        tmp = min(batch_size, x.shape[1] - i)
        b = x[0, i: i + tmp] # get one batch b.shape = (tmp, d)
        b = np.expand_dims(b, 1) # b.shape = (tmp, 1, d)
        d = b - x # d.shape = (tmp, n, d)
        dist = euclidean_metric(d) # dist.shape = (tmp, n)
        for j in range(tmp):
            for k in range(j + i, num):
                if dist[j, k] < r:
                    label.union(j + i, k)
        i += tmp

    label.shrink()
    center = np.zeros([label.ntree, dim])
    cnt = np.zeros([label.ntree, 1], int)
    for i in range(num):
        l = label.classidx[i]
        center[l] += x[0,i]
        cnt[l, 0] += 1
    center /= cnt

    return label, center, cnt

def main():
    random.seed(23333)
    imgs, other_infos = utils.deserialize("../data", fontList)
    imgs_shuffled = []
    for i in imgs:
        indices = [j for j in range(i.shape[0])]
        random.shuffle(indices)
        imgs_shuffled.append(i[indices])
    train = np.array([utils.daisy(img) for i in imgs_shuffled for img in i[:len(i) * 4 // 5]])
    train = np.reshape(train, [train.shape[0], -1])
    train_label = np.array([i for i, _ in enumerate(imgs) for j in _[:len(_) * 4 // 5]])
    test = np.array([utils.daisy(img) for i in imgs_shuffled for img in i[len(i) * 4 // 5:]])
    test = np.reshape(test, [test.shape[0], -1])
    test_label = np.array([i for i, _ in enumerate(imgs) for j in _[len(_) * 4 // 5:]])

    print(train.shape)
    
    label, center, cnt = meanshift(train, .7, 20, 4e-2, uniform_weight)
    center_label = np.zeros([label.ntree, len(fontList)], int)
    for i in range(train.shape[0]):
        center_label[label.classidx[label.find(i)], train_label[i]] += 1

    tp = 0
    for idx, t in enumerate(test):
        lst, dist = utils.nearest_neighbour(t, center)
        if np.argmax(center_label[lst[0]]) == test_label[idx]:
            tp += 1
    print('acc =', tp / test.shape[0])

if __name__ == "__main__":
    main()
