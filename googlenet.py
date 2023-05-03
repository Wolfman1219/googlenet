import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
from tensorflow.keras.utils import plot_model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    conv_1x1 = layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)

    conv_3x3_reduce = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3_reduce)

    conv_5x5_reduce = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5_reduce)

    max_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(max_pool)

    output = layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=-1)
    return output

def googlenet(input_shape, num_classes):
    input_layer = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu')(input_layer)
    x = layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
    x = layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = inception_module(x, 128, 128, 192, 32, 96, 64)
    x = layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)
    x = inception_module(x, 192, 96, 208, 16, 48, 64)
    x = inception_module(x, 160, 112, 224, 24, 64, 64)
    x = inception_module(x, 128, 128, 256, 24, 64, 64)
    x = inception_module(x, 112, 144, 288, 32, 64, 64)
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = inception_module(x, 384, 192, 384, 48, 128, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(input_layer, x)
    return model

input_shape = (224, 224, 3)
num_classes = 12
batch_size = 128
def preprocess_data(example):
    image = example['image']
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    label = example['super_class_id']
    return image, label

dataset = tfds.load("stanford_online_products", split = "train")

ds_test = tfds.load("stanford_online_products", split = "test")
# Preprocess the data


dataset = dataset.map(preprocess_data)
ds_test = ds_test.map(preprocess_data)

dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


model = googlenet(input_shape, num_classes)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 100
plot_model(model, to_file='googlenet.png', show_shapes=True)

history = model.fit(
    dataset,
    epochs=epochs,
    validation_data=ds_test,
)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
