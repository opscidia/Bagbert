import re, os
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers as tr
import nlpaug.augmenter.word as naw


from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from typing import Union, Optional, Any, List


class Augmenter:
    def __init__(
        self,
        model: Optional[str] = None
    ) -> None:
        self.context = naw.ContextualWordEmbsAug(
            model_path = model,
            action = "substitute",
            aug_max = None, aug_p = .5) if model else None
        self.synonym = naw.SynonymAug(
            aug_src = 'wordnet',
            aug_max = None, aug_p = .75)
        self.majority = [ # arbitrary list as a result of an exploratory analysis
            'Prevention', 'Case Report', 'Mechanism; Treatment',
            'Diagnosis', 'Treatment', 'Diagnosis; Treatment',
            'Prevention; Transmission'
        ]
    

    def _augment(self, x):
        x = self.context.augment(x) if self.context else x
        x = self.synonym.augment(x)
        return x

    def augment(self, path: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(path, str): data = pd.read_csv(path).fillna('')
        else: data = path.fillna('')

        additional = data.query(f"label not in {self.majority}")
        with tqdm(total=1, position=0, leave=True) as pbar:
            tqdm.pandas()
            additional['abstract'] = additional.abstract.progress_apply(
                lambda x:self._augment(x))
        return pd.concat([data, additional]).reset_index(drop = True)



def prepare_dataset(
    path: Union[str, pd.DataFrame],
    mode: str = 'tak',
    clean: Union[bool, str] = True,
    tokenizer: Optional[tr.AutoTokenizer] = None,
    only_x: bool = False,
    only_y: bool = False,
    only_text: bool = False,
    **kwargs: Any
) -> Union[tf.Tensor, tf.data.Dataset, None]:
    """
    Create dataset

    :path: path to dataset or pandas dataframe
    :mode: Columns initals. Eg: tak stand for title-abstract-keywords
    :clean: 'mask' or bool. If 'mask', use mask token
    """
    global AUTO
    global REPLICAS

    if isinstance(path, str): data = pd.read_csv(path).fillna('')
    else: data = path.fillna('')

    names = [
        r'covid\s?-?(19)?',
        r'hcov\s?-?(19)?',
        r'sars\s?-?cov\s?-?2?',
        r'sras\s?-?cov\s?-?2?',
        r'sarsr\s?-?cov',
        r'2019\s?-?n\s?-?cov',
        r'corona\s?(virus(es)?)?']

    classes = [
        'Treatment', 'Diagnosis', 'Prevention',
        'Mechanism', 'Transmission',
        'Epidemic Forecasting', 'Case Report']
    

    mlb = MultiLabelBinarizer(classes)
    mlb.fit([classes])

    sub = ' '
    if clean:
        pat = rf"({'|'.join(names)})"
        if clean == 'mask': sub = tokenizer.mask_token
    else: pat = ' '

    clean_ = lambda x: re.sub(r'\( ?\)', '', re.sub(r' +', r' ', re.sub(pat, sub, x, flags=re.I))).strip() if isinstance(x,str) else 'empty'
    clean_k = lambda x:', '.join(re.sub(r' +', r' ', y.strip()) for y in re.split(r';|,', re.sub(pat, sub, x, flags=re.I).strip()) if y.strip() != '') if isinstance(x,str) else ''

    data['keywords'] = data.keywords.apply(clean_k)
    data['title'] = data.title.apply(clean_)
    data['abstract'] = data.abstract.apply(clean_)

    mode_dict = {'t': 'title', 'a': 'abstract', 'k': 'keywords'}
    col = [mode_dict.get(m) for m in mode]

    try:
        y = tf.convert_to_tensor(
            np.array(data.label.apply(
                lambda x:mlb.transform([x.split(';')])[0]).to_list()
            ), tf.int32)
    except:
        print('Warning: Missing labels. This message is not expected if you plan to train')
        y = None
    if only_y: return y
    if only_text: return data
    data['keywords'] = data['keywords'].apply(lambda x: "keywords: "+x)
    data['abstract'] = data['abstract'].apply(lambda x: "text: "+x)
    data['title'] = data['title'].apply(lambda x: "title: "+x)
    x = data[col].apply(lambda x:  re.sub(r' +', ' ', f" {tokenizer.sep_token} ".join([x.iloc[i] for i in range(len(col))])), axis = 1).to_list()
    x = tokenizer.batch_encode_plus(
        x,
        padding = 'max_length',
        truncation = True,
        max_length = kwargs.get('maxlen', 350),
        return_attention_mask = False,
        return_tensors = 'tf')['input_ids']
    if only_x: return x

    auto = globals().get('AUTO', tf.data.experimental.AUTOTUNE)
    replicas = int(globals().get('REPLICAS', 1))

    data = (
        tf.data.Dataset
        .from_tensor_slices((x,y) if y is not None else x)
        .shuffle(int(kwargs.get('buffer', 1_000)))
        .batch(int(kwargs.get('batch', 24*replicas)))
        .prefetch(auto)
    )
    return data



def create_samples(
    path: Union[str, pd.DataFrame],
    output: str,
    modes: Union[str, List[str]],
    fields: Optional[Union[str, List[str]]] = None,
    augment_model: Augmenter = None
):
    """
    Create csv samples
    path: train dataset path or dataframe
    output: output dir
    modes: list or str, in ['fields', 'mask', 'augment'] or 'all'
    fielts: list or str. Dafault, 'tak'
    augment_model: Augmenter
    """
    modes_ = ['fields', 'mask', 'augment', 'all']
    fields_ = ['tak', 'tka', 'all']
    log = '[+] Creating {} task.'
    if modes == 'all': modes = modes_[:-1]
    if fields == 'all': fields = fields_[:-1]
    elif isinstance(fields, str): fields = [fields]
    elif fields is None: fields = ['tak']
    if 'mask' in modes: mask = [True, False]
    else: mask = [False]

    if 'augment' in modes:
        print(log.format('augment'))
        paths = [path, augment_model.augment(path)]
        augment = ('raw', 'augmented')
    else: paths, augment = [path], ('raw',)
    for field in fields:
        pout = f"dataset_{field}"
        for p, a in zip(paths, augment):
            pout = f"{pout}_{a}"
            for m in mask:
                print(log.format(f'{field}, mask {m}, {a}'))
                prepare_dataset(
                    p, field, m, None,
                    only_text = True
                ).to_csv(
                    os.path.join(output, f"{pout}_{str(m)}.csv"),
                    index = False
                )

