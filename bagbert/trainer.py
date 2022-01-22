import tensorflow as tf
import tensorflow_addons as tfa
import transformers as tr
import os, shutil
from tqdm import tqdm

from glob import glob
from sklearn.preprocessing import MinMaxScaler
from dataset import prepare_dataset
from typing import Union, List, Optional
from model import (
    MetaModel,
    SaveCheckpoint
)



def select(
    model_path: str,
    k: int,
    copy: bool = True
) -> Union[None, List[str]]:
    """
    Select k epochs
    """
    ckpts = glob(os.path.join(model_path, 'checkpoints/*'))
    selected = sorted(ckpts, key = lambda x:x.split('hamming-')[1])[:k]
    if not copy: return selected
    for i, ckpt in enumerate(selected):
        subname = f"selected/submodel-{(i+1)}-ckpt{ckpt.split('ckpt')[1]}"
        shutil.copytree(ckpt, os.path.join(model_path, subname))


def weighted_select(
    exp_path: str,
    min_k: int = 0,
    max_k: int = 5
) -> None:
    """
    Create selected/ folder for each model in exp_path/
    """
    scaler = MinMaxScaler((min_k, max_k))
    models_paths = glob(os.path.join(exp_path, '*'))
    tops = [
        [1-float(top[0].split('hamming-')[1])] if top else [None]
        for top in list(map(lambda x:select(x, 1, False), models_paths))
    ]
    weights = list(map(
        lambda x:x if min_k<=x<=max_k else 0,
        (scaler.fit_transform(tops).round()
        .flatten().astype(int).tolist())
    ))
    for k, model in tqdm(zip(weights, models_paths), total = len(weights)):
      select(model, k, True)


def train(
    model_path: str,
    train_path: str,
    val_path: str,
    fields: str,
    clean: bool,
    strategy: tf.distribute.Strategy,
    epochs: int = 1000
):
    global DONE
    DONE = 0

    model_path = model_path if model_path[-1] != '/' else model_path[:-1]
    conf = glob(os.path.join(model_path, 'config.json'))
    assert len(conf), "model_path do not contain model config"
    os.makedirs(os.path.join(model_path, "checkpoints/"), exist_ok=True)

    classes = [
        'Treatment', 'Diagnosis', 'Prevention',
        'Mechanism', 'Transmission',
        'Epidemic Forecasting', 'Case Report']

    with strategy.scope():
        model = MetaModel.from_pretrained(model_path)
        tokenizer = tr.AutoTokenizer.from_pretrained(model_path)

        model.setconfig(fields, clean, classes)

        train_data = prepare_dataset(
            train_path, mode = fields,
            clean = clean,
            tokenizer = tokenizer)

        val_data = prepare_dataset(
            val_path, mode = fields,
            clean = False,
            tokenizer = tokenizer)
        
        early = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss', patience = 10,
            restore_best_weights = True
        )

        reduce = tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss', factor = 0.2, patience = 3,
            verbose = 1, mode = 'auto',
            min_delta = 1e-4, cooldown = 0, min_lr = 1e-9
        )

        save = SaveCheckpoint()
        loss = tf.losses.BinaryCrossentropy(from_logits = True, name = 'loss')
        metrics = [tfa.metrics.HammingLoss(
            mode = 'multilabel',
            threshold = 0.5,
            name = 'hamming'
        )]

        callbacks = [save, early, reduce]

        optimizers = [
            tf.keras.optimizers.Adam(x)
            for x in (3e-6, 1e-9)
        ]
    
    model.compile(optimizer=optimizers[0], metrics=metrics, loss=loss)
    model.summary()

    print('[+] Training')
    epochs_done = DONE
    history = model.fit(
        train_data, epochs = epochs,
        callbacks = callbacks, 
        validation_data = val_data,
        initial_epoch = epochs_done
    )

    print('[+] Saving optimal weights')
    model.save_pretrained(os.path.join(model_path, 'optimal/'))
    tokenizer.save_pretrained(os.path.join(model_path, 'optimal/'))

