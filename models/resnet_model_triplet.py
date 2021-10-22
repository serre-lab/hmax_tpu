import tensorflow as tf
from tensorflow.keras.layers import *

def get_triplet_model(input_shape = (600, 600, 3),
                      embedding_units = 256,
                      embedding_depth = 2,
                      backbone_class=tf.keras.applications.ResNet50V2,
                      nb_classes = 19):

    backbone = backbone_class(input_shape=input_shape, include_top=False)
    features = GlobalAveragePooling2D()(backbone.output)

    embedding_head = features
    for embed_i in range(embedding_depth):
        embedding_head = Dense(embedding_units, activation="relu" if embed_i < embedding_depth-1 else "linear")(embedding_head)
    #embedding_head = tf.nn.l2_normalize(embedding_head, -1, epsilon=1e-5,name='embeddings')
    embedding_head = Lambda(lambda x:tf.nn.l2_normalize(x, -1, epsilon=1e-5) ,name='embedding')(embedding_head)
    logits_head = Dense(nb_classes,name='logits')(features)

    model = tf.keras.Model(backbone.input, [embedding_head, logits_head])
    model.compile(loss='cce')
    #model.summary()

    return model