import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import net_def as nd
import utils
import random
import argparse
from summary import summarizer
import os

fontList = ['CALIFORNIAN', 'HARRINGTON', 'BRUSH', 'MODERN', 'PAPYRUS', 'EDWARDIAN', 'FREESTYLE']
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
log_file_name = 'log.csv'
batch_size = 16

def dataset(Dir, fontList = None):
    imgs, other_info = utils.deserialize(Dir)#, fontList)
    labels = np.concatenate(
        [np.ones(i.shape[0], np.int32) * k for k, i in enumerate(imgs)],
        0
    )
    imgs = np.concatenate(imgs, 0).astype(np.float32) / 255
    imgs = np.expand_dims(imgs, -1)

    random.seed(23333)
    indices = [i for i in range(imgs.shape[0])]
    random.shuffle(indices)
    imgs = imgs[indices]
    labels = labels[indices]
    num_train = imgs.shape[0] * 3 // 5
    num_valid = imgs.shape[0] // 5

    train = tf.data.Dataset.from_tensor_slices(
        (imgs[:num_train], labels[:num_train])
    ).map(
        lambda img, label: (tf.image.resize_image_with_pad(
            tf.random_crop(img, [48, 48, 1]),
            64, 64
        ), label)
    ).shuffle(256).repeat().batch(batch_size)
    valid = tf.data.Dataset.from_tensor_slices((imgs[num_train:num_train+num_valid], labels[num_train:num_train+num_valid])).repeat().batch(batch_size)
    test = tf.data.Dataset.from_tensor_slices((imgs[num_train+num_valid:], labels[num_train+num_valid:])).batch(batch_size)

    return train, valid, test, imgs.shape[0]*3//5

def datasetV2(Dir):
    # names = os.listdir(Dir)
    # names = list(map(lambda x: os.path.splitext(x)[0], names))
    # names = np.unique(names)
    names = fontList
    names = list(map(lambda x: os.path.join(Dir, x + '.npy'), names))

    num_train = 0
    for n in names:
        num_train += np.load(n).shape[0]*3//5

    def process(name, label, tag):
        data = np.load(name.decode()).astype(np.float32) / 255
        data = np.expand_dims(data, -1)
        random.seed(23333)
        indices = [i for i in range(data.shape[0])]
        random.shuffle(indices)
        
        if tag.decode() == "train":
            indices = indices[:data.shape[0]*3//5]
        elif tag.decode() == "valid":
            indices = indices[data.shape[0]*3//5:data.shape[0]*4//5]
        else:
            indices = indices[data.shape[0]*4//5:]

        data = data[indices]
        
        # if np.any(np.isnan(data)):
        #     print(name, data.shape, label, tag)
        return data, np.ones([data.shape[0]], np.int32) * label

    ds = tf.data.Dataset.from_tensor_slices((names, list([i for i in range(len(names))])))

    train = ds.flat_map(
        lambda name, label: tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_func(process, [name, label, 'train'], [tf.float32, tf.int32]))
        )
    ).repeat().shuffle(256).batch(batch_size)

    valid = ds.flat_map(
        lambda name, label: tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_func(process, [name, label, 'valid'], [tf.float32, tf.int32]))
        )
    ).repeat().batch(batch_size)

    test = ds.flat_map(
        lambda name, label: tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_func(process, [name, label, 'test'], [tf.float32, tf.int32]))
        )
    ).batch(batch_size)

    return train, valid, test, num_train

def prepare(datadir):
    utils.readCSVAndSerialize('../fonts', datadir
    , list(map(lambda x:x+'.csv', fontList))
    , filt=lambda x:int(x["m_label"])<256
    )

def main(datadir, model, ckpt, restore = True, dotest = False):
    train, valid, test, num_train = dataset(datadir)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train.output_types,
        (tf.TensorShape([None, 64, 64, 1]), tf.TensorShape([None]))
    )
    steps = tf.Variable(0, dtype = tf.int64, name = "global_step", trainable = False)

    x, label = iterator.get_next()
    y = model(x, len(fontList))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(label),
        logits=y)
    mloss = tf.reduce_mean(loss)
    acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(y, -1, output_type=tf.int32),
                label
            ),
            tf.float32
        )
    )
    train_op = tf.train.AdamOptimizer(2e-4).minimize(loss, global_step=steps)

    train_iter = train.make_one_shot_iterator()
    valid_iter = valid.make_one_shot_iterator()
    test_iter = test.make_one_shot_iterator()

    config = tf.ConfigProto() 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.allow_soft_placement=True
    
    # if restore:
    #     reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
    #     var_to_shape_map = reader.get_variable_to_shape_map()
    #     restore_dict = dict()
    #     for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #         tname = v.name.split(':')[0]
    #         if reader.has_tensor(tname) and var_to_shape_map[tname] == v.shape:
    #             restore_dict[tname] = v
    #     restorer = tf.train.Saver(restore_dict)

    saver = tf.train.Saver()

    if not dotest:
        summary = summarizer(
            os.path.join('../log', log_file_name),
            ['step', 'epoc', 'train_loss', 'val_loss', 'train_acc', 'val_acc'],
            32, restore = restore
        )

    with tf.Session(config=config) as sess:
        train_hdl = sess.run(train_iter.string_handle())
        valid_hdl = sess.run(valid_iter.string_handle())
        test_hdl = sess.run(test_iter.string_handle())

        print('restoring/initializing')
        
        if restore:
            sess.run(tf.global_variables_initializer())
            # restorer.restore(sess, ckpt)
            saver.restore(sess, ckpt)
            print('restored')
        else:
            sess.run(tf.global_variables_initializer())
            print('initial done')
        
        if dotest:
            tacc = 0.
            cnt = 0
            while True:
                try:
                    output, _acc = sess.run([y, acc], feed_dict={handle: test_hdl})
                    tacc += _acc * output.shape[0]
                    cnt += output.shape[0]
                    print(output.shape[0], _acc)
                except tf.errors.OutOfRangeError:
                    print(cnt, tacc / cnt)
                    break
            return

            
        cnt = epoc = 0
        while epoc < 400:
            _, _loss, _acc = sess.run([train_op, mloss, acc], feed_dict={handle: train_hdl})
            cnt += batch_size
            if summary.step == summary.steps - 1:
                vl, va = sess.run([mloss, acc], feed_dict = {handle: valid_hdl})
                summary(
                    train_loss = _loss,
                    val_loss = vl,
                    train_acc = _acc,
                    val_acc = va,
                    step = sess.run(steps),
                    epoc = epoc
                )
            else:
                summary(train_loss = _loss, train_acc = _acc)
            
            if cnt >= num_train:
                epoc += 1
                cnt -= num_train
                print('epoc %d done'%epoc)
                if epoc % 5 == 0:
                    saver.save(sess, ckpt)
                    print('model saved')
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='../data-dnn', help='Directory of data, default to ../data-dnn')
    parser.add_argument('-i', '--init', action='store_true', help='Initialize model instead using of checkpoint (default not)')
    parser.add_argument('-c', '--ckpt', type=str, help='Directory to store checkpoint', required=True)
    parser.add_argument('-m', '--model', type=str, choices=['resnet', 'lenet'], default='resnet', help='The model to use (resnet/lenet)')
    parser.add_argument('-P', '--prepare', action='store_true', help='Prepare datafile for DNN')
    parser.add_argument('-T', '--test', action='store_true', help='Do test')

    args = parser.parse_args()

    if args.prepare:
        prepare(args.dir)
        quit()

    if args.model == 'resnet':
        model = nd.resnet_18
    elif args.model == 'lenet':
        model = nd.lenet
    log_file_name = 'log_%s.csv'%args.model

    main(args.dir, model, args.ckpt, not args.init, args.test)
    