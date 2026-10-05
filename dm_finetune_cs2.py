# Finetune the pretrained BC agent on CS2 self-recorded data
# (Stage 3 of docs/CS2迁移测试计划.md).
#
# Pipeline: dm_record_data_cs2.py *.npz  ->  this script  ->  stateful model
#
# What it does:
#   1. loads all cs2_ft_*.npz from --data-dir (imgs + keys + clicks + mouse)
#   2. builds (96-frame sequence -> 52-dim action) samples exactly like the
#      original DataGenerator: mouse bucket quantisation, extreme-class remap,
#      left-click gap fill, brightness/contrast augmentation
#   3. loads the pretrained base model (non-stateful json+h5) and finetunes
#      with the original custom loss (keys/clicks BCE + mouse CE + critic)
#   4. saves the finetuned model AND a stateful inference twin
#
# Naming: the finetuned model keeps 'drop' in its name so every downstream
# name-flag code path (stateful twin) rebuilds the same architecture.
#
# Usage (csgo_tf210 env):
#   python dm_finetune_cs2.py                          # full finetune
#   python dm_finetune_cs2.py --epochs 1 --max-files 2   # quick trial

import os
import sys
import glob
import argparse

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy(['GPU:0'])

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Flatten, LSTM,
                                     ConvLSTM2D, Input, concatenate,
                                     TimeDistributed)
from tensorflow.keras import optimizers, losses
import tensorflow.keras.backend as K
from tensorflow.keras.applications import EfficientNetB0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (csgo_img_dimension, N_TIMESTEPS, n_keys, n_clicks,
                    n_mouse_x, n_mouse_y, mouse_x_possibles, mouse_y_possibles,
                    mouse_x_lim, mouse_y_lim, mouse_preprocess,
                    actions_to_onehot, tp_load_model, tp_save_model, GAMMA)

# ----------------------------------------------------------------- settings
def parse_args():
    ap = argparse.ArgumentParser(description='Finetune BC agent on CS2 data')
    ap.add_argument('--data-dir', default=r'D:\csgo_data\cs2_ft')
    ap.add_argument('--base-model', default='ak47_sub_55k_drop_d4',
                    help='pretrained non-stateful model in ./model/')
    ap.add_argument('--out-name', default='cs2ft_drop_d1',
                    help='finetuned model name (keep "drop" in it!)')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--l-rate', type=float, default=1e-4)
    ap.add_argument('--max-files', type=int, default=0,
                    help='use only the first N .npz files (0 = all)')
    ap.add_argument('--skip-train', action='store_true',
                    help='skip training, load the saved finetuned model and '
                         'only (re)build the stateful twin')
    return ap.parse_args()


ARGS = args = parse_args()

# ------------------------------------------------------------ load .npz data
def load_dataset(folder, max_files):
    files = sorted(glob.glob(os.path.join(folder, 'cs2_ft_*.npz')),
                   key=lambda p: int(os.path.basename(p)[7:-4]))
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit('no cs2_ft_*.npz found in %s' % folder)
    imgs_l, keys_l, clicks_l, mouse_l = [], [], [], []
    for p in files:
        z = np.load(p)
        if 'mouse' not in z:
            print('skip (v1, no mouse):', os.path.basename(p))
            continue
        imgs_l.append(z['imgs'])
        keys_l.append(z['keys'])
        clicks_l.append(z['clicks'])
        mouse_l.append(z['mouse'])
    imgs = np.concatenate(imgs_l)
    keys = np.concatenate(keys_l)
    clicks = np.concatenate(clicks_l)
    mouse = np.concatenate(mouse_l).astype(np.float64)
    print('dataset: %d files, %d frames (~%.1f min) | fire %.2f%% | '
          'mouse-delta nonzero %.0f%%'
          % (len(files), len(imgs), len(imgs) / 16 / 60,
             100 * clicks[:, 0].mean(),
             100 * ((np.abs(mouse[:, 0]) + np.abs(mouse[:, 1])) > 0).mean()))
    return imgs, keys, clicks, mouse


# --------------------------------------------------- labels (original logic)
N_JITTER = 20
ACTIONS_DIM = n_keys + n_clicks + n_mouse_x + n_mouse_y  # 51


def build_labels(keys_onehot, clicks, mouse):
    """Per-frame 51-dim target + the original generator's clean-up hacks."""
    n = len(keys_onehot)
    y = np.zeros((n, ACTIONS_DIM), dtype=np.float64)
    for i in range(n):
        mx, my = mouse_preprocess(mouse[i, 0], mouse[i, 1])
        mx_id = mouse_x_possibles.index(mx)
        my_id = mouse_y_possibles.index(my)
        y[i, 0:n_keys] = keys_onehot[i]
        y[i, n_keys:n_keys + n_clicks] = clicks[i]
        y[i, n_keys + n_clicks + mx_id] = 1
        y[i, n_keys + n_clicks + n_mouse_x + my_id] = 1

    # extreme mouse classes remapped to neighbours (as in original generator)
    for i in range(n):
        base = n_keys + n_clicks
        if y[i, base] == 1:
            y[i, base] = 0; y[i, base + 2] = 1
        elif y[i, base + 1] == 1:
            y[i, base + 1] = 0; y[i, base + 2] = 1
        elif y[i, base + n_mouse_x - 1] == 1:
            y[i, base + n_mouse_x - 1] = 0; y[i, base + n_mouse_x - 3] = 1
        elif y[i, base + n_mouse_x - 2] == 1:
            y[i, base + n_mouse_x - 2] = 0; y[i, base + n_mouse_x - 3] = 1
        base_y = base + n_mouse_x
        if y[i, base_y] == 1:
            y[i, base_y] = 0; y[i, base_y + 2] = 1
        elif y[i, base_y + 1] == 1:
            y[i, base_y + 1] = 0; y[i, base_y + 2] = 1
        elif y[i, base_y + n_mouse_y - 1] == 1:
            y[i, base_y + n_mouse_y - 1] = 0; y[i, base_y + n_mouse_y - 3] = 1
        elif y[i, base_y + n_mouse_y - 2] == 1:
            y[i, base_y + n_mouse_y - 2] = 0; y[i, base_y + n_mouse_y - 3] = 1

    # fill single-frame gaps in left-click (fire-rate < frame-rate hack)
    for j in range(1, n - 1):
        if y[j - 1, n_keys] == 1 and y[j + 1, n_keys] == 1:
            y[j, n_keys] = 1
    for j in range(1, n - 2):
        if y[j - 1, n_keys] == 1 and y[j + 2, n_keys] == 1:
            y[j, n_keys] = 1
            y[j + 1, n_keys] = 1
    return y


# ------------------------------------------------------- loss and metrics
def custom_loss(y_true, y_pred):
    loss1a = losses.binary_crossentropy(y_true[:, :, 0:4], y_pred[:, :, 0:4])
    loss1b = losses.binary_crossentropy(y_true[:, :, 4:5], y_pred[:, :, 4:5])
    loss1c = losses.binary_crossentropy(y_true[:, :, n_keys - 1:n_keys],
                                        y_pred[:, :, n_keys - 1:n_keys])
    loss1d = losses.binary_crossentropy(y_true[:, :, n_keys - 4:n_keys - 1],
                                        y_pred[:, :, n_keys - 4:n_keys - 1])
    loss2a = losses.binary_crossentropy(y_true[:, :, n_keys:n_keys + 1],
                                        y_pred[:, :, n_keys:n_keys + 1])
    loss3 = losses.categorical_crossentropy(
        y_true[:, :, n_keys + n_clicks:n_keys + n_clicks + n_mouse_x],
        y_pred[:, :, n_keys + n_clicks:n_keys + n_clicks + n_mouse_x])
    loss4 = losses.categorical_crossentropy(
        y_true[:, :, n_keys + n_clicks + n_mouse_x:
               n_keys + n_clicks + n_mouse_x + n_mouse_y],
        y_pred[:, :, n_keys + n_clicks + n_mouse_x:
               n_keys + n_clicks + n_mouse_x + n_mouse_y])
    # critic loss with trivial rewards (no kill labels in v2 data)
    loss_crit = 10 * losses.MSE(
        y_true[:, :, ACTIONS_DIM:ACTIONS_DIM + 1],
        y_pred[:, :, ACTIONS_DIM:ACTIONS_DIM + 1])
    return K.concatenate([loss1a, loss1b, loss1c, loss1d, loss2a, loss3,
                          loss4, loss_crit])


def wasd_acc(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(y_true[:, :, 0:4], y_pred[:, :, 0:4])

def Lclk_acc(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(y_true[:, :, n_keys:n_keys + 1],
                                            y_pred[:, :, n_keys:n_keys + 1],
                                            threshold=0.5)

def m_x_acc(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(
        y_true[:, :, n_keys + n_clicks:n_keys + n_clicks + n_mouse_x],
        y_pred[:, :, n_keys + n_clicks:n_keys + n_clicks + n_mouse_x])

def m_y_acc(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(
        y_true[:, :, n_keys + n_clicks + n_mouse_x:
               n_keys + n_clicks + n_mouse_x + n_mouse_y],
        y_pred[:, :, n_keys + n_clicks + n_mouse_x:
               n_keys + n_clicks + n_mouse_x + n_mouse_y])


# ------------------------------------------------------------- data generator
class NpzDataGenerator(tf.keras.utils.Sequence):
    """(96-frame sequence) -> 53-col target, mirroring the original generator."""

    def __init__(self, imgs, keys, clicks, mouse, batch_size=1, shuffle=True):
        self.imgs, self.keys, self.clicks, self.mouse = imgs, keys, clicks, mouse
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_frames = len(imgs)
        n_seq = (self.n_frames - N_TIMESTEPS) // N_TIMESTEPS
        self.starts = np.arange(n_seq) * N_TIMESTEPS
        self.on_epoch_end()
        print('generator: %d sequences of %d frames' % (len(self.starts), N_TIMESTEPS))

    def __len__(self):
        return len(self.starts)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.starts)

    def _y_target(self, s):
        y = build_labels(self.keys[s:s + N_TIMESTEPS],
                         self.clicks[s:s + N_TIMESTEPS],
                         self.mouse[s:s + N_TIMESTEPS])
        out = np.zeros((N_TIMESTEPS, ACTIONS_DIM + 2))  # +reward, +advantage
        out[:, :-2] = y
        out[:, -2] = -0.01 * y[:, n_keys]  # reward = kill -0.5*death -0.01*shoot
        return out

    def __getitem__(self, index):
        s = int(self.starts[index])
        if s + N_TIMESTEPS + N_JITTER < self.n_frames:
            s += np.random.randint(0, N_JITTER)
        X = self.imgs[s:s + N_TIMESTEPS].astype(np.float64)
        y = self._y_target(s)

        # brightness / contrast augmentation (as in original)
        if np.random.rand() < 0.5:
            bright = np.random.rand() * 0.6 + 0.7
            X = np.clip(X * bright, 0, 255)
        if np.random.rand() < 0.5:
            contrast = np.random.rand() * 0.6 + 0.7
            X = np.clip(128 + contrast * X - contrast * 128, 0, 255)

        X = np.expand_dims(X, 0)
        y = np.expand_dims(y, 0)
        return X, y


# --------------------------------------------------------- stateful twin
def create_stateful(model, model_name, save_dir):
    """Rebuild the model with stateful ConvLSTM/LSTM and copy weights."""
    input_shape_batch = (1, 1, csgo_img_dimension[0], csgo_img_dimension[1], 3)
    base_model = EfficientNetB0(weights='imagenet',
                                input_shape=(csgo_img_dimension[0],
                                             csgo_img_dimension[1], 3),
                                include_top=False, drop_connect_rate=0.2)
    intermediate_model = Model(inputs=base_model.input,
                               outputs=base_model.layers[161].output)
    input_1 = Input(batch_shape=input_shape_batch, name='main_in')
    x = TimeDistributed(intermediate_model)(input_1)
    x = ConvLSTM2D(filters=256, kernel_size=(3, 3), stateful=True,
                   return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(x)
    x = TimeDistributed(Flatten())(x)
    output_1 = Dense(n_keys, activation='sigmoid')(x)
    output_2 = Dense(n_clicks, activation='sigmoid')(x)
    output_3 = Dense(n_mouse_x, activation='softmax')(x)
    output_4 = Dense(n_mouse_y, activation='softmax')(x)
    output_5 = Dense(1, activation='linear')(x)
    model_stateful = Model(input_1, output_all := concatenate(
        [output_1, output_2, output_3, output_4, output_5], axis=-1))
    # no compile here: the twin is inference-only, and compiling with custom
    # metrics makes model.to_json() fail on TF 2.10 (EagerTensor serialization)
    for nb, layer in enumerate(model.layers):
        model_stateful.layers[nb].set_weights(layer.get_weights())

    # TF 2.10 to_json() cannot serialize EfficientNetB0's BN eager stats, so
    # we cannot tp_save_model the twin directly. Instead: reuse the ORIGINAL
    # pretrained stateful JSON as the architecture template (TF 2.3 saved it
    # cleanly and its layer order is identical), and only store fresh weights.
    import json
    import pickle
    template_path = os.path.join(save_dir, ARGS.base_model + '_stateful.json')
    arch = json.load(open(template_path))
    out_json = os.path.join(save_dir, model_name + '_stateful.json')
    json.dump(arch, open(out_json, 'w'))
    print('saved model to ', out_json)

    out_h5 = os.path.join(save_dir, model_name + '_stateful.h5')
    model_stateful.save_weights(out_h5)
    print('saved weights to ', out_h5)

    hypers_path = os.path.join(save_dir, ARGS.base_model + '_stateful.p')
    if os.path.isfile(hypers_path):
        import shutil
        shutil.copy(hypers_path,
                    os.path.join(save_dir, model_name + '_stateful.p'))
        print('copied hypers from', os.path.basename(hypers_path))
    return model_stateful


# --------------------------------------------------------------------- main
def main():
    with strategy.scope():
        if ARGS.skip_train:
            print('-- loading finetuned model: %s --' % ARGS.out_name)
            model = tp_load_model('model', ARGS.out_name)
        else:
            imgs, keys, clicks, mouse = load_dataset(ARGS.data_dir, ARGS.max_files)
            print('-- loading pretrained base model: %s --' % ARGS.base_model)
            model = tp_load_model('model', ARGS.base_model)
            opt = optimizers.Adam(learning_rate=ARGS.l_rate)
            model.compile(loss=custom_loss, optimizer=opt,
                          metrics=[Lclk_acc, m_x_acc, m_y_acc, wasd_acc])
            print('model loaded & compiled')

            gen = NpzDataGenerator(imgs, keys, clicks, mouse, batch_size=1)

            print('-- finetuning: %d epochs --' % ARGS.epochs)
            hist = model.fit(gen, epochs=ARGS.epochs, verbose=1)

        if not ARGS.skip_train:
            save_dir = 'model'
            tp_save_model(model, save_dir, ARGS.out_name)

    print('-- creating stateful twin --')
    create_stateful(model, ARGS.out_name, 'model')
    print('DONE. finetuned model: %s / %s_stateful' % (ARGS.out_name,
                                                       ARGS.out_name))


if __name__ == '__main__':
    main()
