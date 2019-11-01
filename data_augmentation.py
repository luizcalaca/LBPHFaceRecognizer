from keras.preprocessing.image import ImageDataGenerator
import argparse

data_dir = "datasets/train"
img_width, img_height = 224, 224
batch_size = 32

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store augmentation examples")
args = vars(ap.parse_args())

datagen_config = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    shuffle=True,
    seed=42)

generator = datagen_config.flow_from_directory(
    data_dir,
    save_to_dir=args["output"],
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical')
