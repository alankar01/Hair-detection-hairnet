import argparse
from keras.callbacks import ModelCheckpoint
import keras
from keras.optimizers import Adam

from data.load_data import trainGenerator
from nets import Hairnet

import os


def get_args():
    parser = argparse.ArgumentParser(description='Hair Segmentation')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--img_size', type=int, default=256)

    parser.add_argument('--path_model', default='models/hair.hdf5')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    # Augmentation Data
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    size_data = len(os.listdir(f"{args.data_dir}/images"))
    steps_per_epoch = size_data // args.batch_size
    args.steps_per_epoch = steps_per_epoch

    myGene = trainGenerator(args.batch_size, args.data_dir, 'images', 'masks', data_gen_args, save_to_dir=None)
    print(args)
    model = Hairnet.get_model()

    model.compile(optimizer=Adam(lr=args.lr), loss='binary_crossentropy', metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint('models/hairnet_matting.hdf5', monitor='loss', verbose=0, save_best_only=True)
    model.fit_generator(myGene, callbacks=[model_checkpoint], steps_per_epoch=110, epochs=5)

    model.save(args.path_model)
