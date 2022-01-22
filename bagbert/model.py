import os, json
import pandas as pd
import transformers as tr
import tensorflow as tf
from tensorflow.keras import layers
from transformers.modeling_tf_utils import TFSequenceClassificationLoss
from glob import glob
from pprint import pprint as cat

from typing import List, Optional, Any

from dataset import prepare_dataset

# MODELS

class BaseTopic:
    """
    Base Model for Topic Classification
    Using CLS token representation (not pooled)
    """
    def call(
        self,
        inputs = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        labels = None,
        training = False,
    ):
        outputs = self.transformer(
            inputs,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            training = training,
        )
        sequence_output = self.dropout(outputs[0], training = training)
        cls_token = self.stride(sequence_output)
        return self.classifier(cls_token)
    

    def setconfig(
        self,
        fields: str,
        clean: bool,
        classes: List[str]
    ) -> None:
        """
        Save model sample config
        fileds: 'tak', 'tka', 'kta', 'a', ...
        clean: if True, sample does not contain terms related to COVID-19
        classes: list of (7) classes.
        """
        id2label = dict(zip(map(str, range(len(classes))), classes))
        label2id = dict(zip(classes, range(len(classes))))
        setattr(self.config, 'id2label', id2label)
        setattr(self.config, 'label2id', label2id)
        setattr(self.config, 'fields', fields)
        setattr(self.config, 'clean', clean)
    

    def load_data(
        self,
        data,
        tokenizer: Optional[str] = None,
        **kwargs: Any
    ):
        if isinstance(data, (str, pd.DataFrame)):
            only_x = not kwargs.get('training', False)
            data = prepare_dataset(
                data,
                mode = getattr(self.config, 'fields', 'tak'),
                clean = False,
                tokenizer = tokenizer,
                only_x = only_x
            )
        return data
    

    def predict(self, x, *args, **kwargs):
        tokenizer = kwargs.pop('tokenizer', None)
        x = self.load_data(x, tokenizer, **kwargs)
        return super().predict(x, *args, **kwargs)



class BERTTopic(BaseTopic, tr.TFBertPreTrainedModel, TFSequenceClassificationLoss):
    """
    Topic Classification using BERT
    """

    _keys_to_ignore_on_load_missing = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = tr.TFBertMainLayer(config, name = "bert")
        self.dropout = layers.Dropout(
            getattr(config, 'hidden_dropout_prob', .0))
        self.stride = layers.Lambda(lambda x: x[:, 0, :], name = "stride")
        self.classifier = layers.Dense(7, activation = tf.keras.activations.sigmoid, name = "classifier")



class RoBERTaTopic(BaseTopic, tr.TFRobertaPreTrainedModel, TFSequenceClassificationLoss):
    """
    Topic Classification using RoBERTa
    """

    _keys_to_ignore_on_load_missing = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = tr.TFRobertaMainLayer(config, name = "roberta")
        self.dropout = layers.Dropout(
            getattr(config, 'hidden_dropout_prob', .0))
        self.stride = layers.Lambda(lambda x: x[:, 0, :], name = "stride")
        self.classifier = layers.Dense(7, activation = tf.keras.activations.sigmoid, name = "classifier")



class MetaModel(BaseTopic):
    @classmethod
    def from_pretrained(cls, model_path, *args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, tr.PretrainedConfig):
            conf = glob(os.path.join(model_path, 'config.json'))
            if not conf:
                raise Exception(f"No model config found. Is {model_path} a model folder ?")
            with open(conf[0]) as f:
                config = tr.PretrainedConfig.from_dict(json.load(f))
            model_type = getattr(config, 'model_type', 'roberta')
            if model_type == 'bert': model = BERTTopic
            else: model = RoBERTaTopic
            return model.from_pretrained(model_path, *args, **kwargs)
    


class MetaEnsemble(MetaModel):
    
    @classmethod
    def from_pretrained(cls, exp_path, **kwargs):
        model_paths = {m.split(exp_path)[1]:list(map(
            lambda x:x.split('config.json')[0],
            glob(os.path.join(m, 'selected/*/config.json'))
        )) for m in glob(os.path.join(exp_path, '*'))}
        if kwargs.get('verbose', 0):
            for model_path, sub in model_paths.items():
                print(f"{model_path}: {len(sub)} submodel{'s' if len(sub)>1 else ''} selected")
        if not model_paths:
            raise Exception(f"No models config found. Is {exp_path} a models (exp) folder ?")
        
        self = super(MetaEnsemble, cls).__new__(cls)
        self.model_paths = model_paths
        self.exp_path = exp_path
        
        return self
    

    def predict(self, x, *args, **kwargs):
        reload = isinstance(x, (str, pd.DataFrame))
        predictions = list()
        for i, (meta, models) in enumerate(self.model_paths.items()):
            print(f"[+] Model {i+1}/{len(self.model_paths)}: {meta}")
            for model_ in models:
                model = MetaModel.from_pretrained(model_)
                if reload:
                    tokenizer_path = os.path.join(
                        self.exp_path,
                        meta if meta[0] not in ['/', '\\'] else meta[1:])
                    if model.config.model_type.lower() == 'bert':
                        tokenizer = tr.BertTokenizer
                    else: tokenizer = tr.RobertaTokenizer
                    kwargs['tokenizer'] = tokenizer.from_pretrained(tokenizer_path)
                predictions.append(model.predict(x, *args, **kwargs))
        model = None
        return sum(predictions)/len(predictions)
    

    def train(self, *args, **kwargs):
        raise NotImplementedError("MetaEnsemble is not suppposed to train. Only to infer.")

        

# CALLBACKS

class SaveCheckpoint(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=dict()):
    global DONE
    DONE = int(epoch)
    val_loss = logs.get('val_loss', 'nan')
    metric = logs.get('val_hamming', 'nan')
    setattr(self.model.config, 'hamming', metric)
    filepath =  f"{self.model.config._name_or_path}/checkpoints/loss-{val_loss}-ckpt-{epoch}-hamming-{metric}"
    self.model.save_pretrained(filepath)