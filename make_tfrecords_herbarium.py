import os, cv2
import tensorflow as tf
import json
import multiprocessing as mp
#from multiprocessing import Pool, freeze_support

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, img_idx, label):
    feature = {
        "image": image_feature(image),
        "image_idx": int64_feature(img_idx),
        "label" : int64_feature(label)
    }
    #feature = {
    #    "image": image_feature(image),
    #    "path": bytes_feature(path),
    #    "area": float_feature(example["area"]),
    #    "bbox": float_feature_list(example["bbox"]),
    #    "category_id": int64_feature(example["category_id"]),
    #    "id": int64_feature(example["id"]),
    #    "image_id": int64_feature(example["image_id"]),
    #}
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_unlabeled_example(image, image_idx):
    feature = {
        "image": image_feature(image),
        "image_idx": int64_feature(image_idx),
        
    }
    #feature = {
    #    "image": image_feature(image),
    #    "path": bytes_feature(path),
    #    "area": float_feature(example["area"]),
    #    "bbox": float_feature_list(example["bbox"]),
    #    "category_id": int64_feature(example["category_id"]),
    #    "id": int64_feature(example["id"]),
    #    "image_id": int64_feature(example["image_id"]),
    #}
    return tf.train.Example(features=tf.train.Features(feature=feature))




def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_idx": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
        
        
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    return example

def make_tfrecords(args):
    tfrecords_dir,tfrec_num,samples,images = args[0],args[1],args[2],args[3]
    print( tfrecords_dir + "/file_%.5i-%i.tfrec" % (tfrec_num, len(samples)))
    with tf.io.TFRecordWriter(
        tfrecords_dir + "/file_%.5i-%i.tfrec" % (tfrec_num, len(samples))
    ) as writer:
        for sample in samples:
            for img in images:
                if sample['id']==img['id']:
                    image_path = f"{parent}/train/{img['file_name']}"
                    image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                    image = tf.image.resize(image,(600,600))
                    image = tf.cast(image, tf.uint8)
                    example = create_example(image, sample['image_id'], sample['category_id'])
                    writer.write(example.SerializeToString())


            
if __name__ == '__main__':

    parent = '/cifs/data/tserre_lrs/projects/prj_fossils/data/raw_data/Herbarium_2021_FGVC8'
    file = '/cifs/data/tserre_lrs/projects/prj_fossils/data/raw_data/Herbarium_2021_FGVC8/train/metadata.json'
    with open(file,'r') as f : 
        data = json.load(f)
    file = '/cifs/data/tserre_lrs/projects/prj_fossils/data/raw_data/Herbarium_2021_FGVC8/test/metadata.json'
    with open(file,'r') as f : 
        test_data = json.load(f)
    annotations = data['annotations']
    images = data['images']
    tfrecords_dir = '/cifs/data/tserre_lrs/projects/prj_fossils/data/raw_data/Herbarium_2021_FGVC8/tfrecords/train_2'
    num_samples = 10000
    num_tfrecords = len(annotations) // num_samples
    if len(annotations) % num_samples:
        num_tfrecords += 1  # add one record if there are any remaining samples

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)  # creating TFRecords output folder
    print('starting mp')
    #pool = mp.Pool(processes=4)
    workers = 8
    #mp.freeze_support()
    for tfrec_num in range(num_tfrecords):
        samples = annotations[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)]
        #pool.map(make_tfrecords,args=(tfrecords_dir,tfrec_num,samples,images))
        p = mp.Process(make_tfrecords,args=(tfrecords_dir,tfrec_num,samples,images))
        p.start()
        #p.join()

        
