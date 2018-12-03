import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import io

key2idx = {"font": 0, "fontVariant": 1, "m_label": 2, "strength": 3, "italic": 4, "m_top": 5, "m_left": 6, "h": 7, "w": 8, "img": 9}

def readCSV(csvPath = '../fonts', origin = False, filt = lambda x: int(x["m_label"]) < 256, fontList = None):
    imgs = []
    other_info = []
    cnt = 0
    if fontList is None:
        fontList = os.listdir(csvPath)

    for i in fontList:
        print("%d/%d" % (cnt, len(fontList)))
        cnt += 1
        if i.split('.')[-1].lower() != 'csv':
            continue
        path = os.path.join(csvPath, i)
        with open(path) as f:
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
                    im_resize = im.resize([ow, oh])
                    img = np.zeros([oh + st * 2, ow + sl * 2], dtype = np.uint8)
                    img[st: st + oh, sl: sl + ow] = np.array(im_resize)
                    
                # plt.figure()
                # plt.imshow(img, cmap='Greys', vmin = 0, vmax = 255)
                # plt.show()
                other_info.append([line["font"], line["fontVariant"], int(line["m_label"]), float(line["strength"]), int(line["italic"]), st, sl, h, w])
                imgs.append(img)
    return np.array(imgs), other_info

def main():
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
    # readCSV(origin = True)
    imgs, infos = readCSV(fontList = ['ARIAL.csv'], filt = lambda x: int(x["m_label"]) < 128)
    d = {}
    for l in infos:
        k = l[1]
        if not k in d:
            d[k] = 1
        else:
            d[k] += 1
    print("num of fontVariant:", d)

if __name__ == '__main__':
    main()