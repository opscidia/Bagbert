# BagBert: BERT-based bagging-stacking for multi-topic classification

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8-3776AB?style=for-the-badge&logo=Python&logoColor=f1c40f)](https://www.python.org/downloads/)
[![Tensorflow](https://img.shields.io/badge/tensorflow-2.4-ff6f00?style=for-the-badge&logo=tensorflow)](https://www.djangoproject.com/download/)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.6322789-3776AB?style=for-the-badge)](https://doi.org/10.5281/zenodo.6322789)

</div>

Code implementation of [BagBert: Bert-based bagging-stacking](https://arxiv.org/abs/2111.05808)


## Requirements
```sh
python -m pip install -r requirements.txt
```

## Commands
**`sample`**: Create CSV samples from training dataset.

```sh
python bagbert sample data/BC7-LitCovid-Train.csv


positional arguments:
  path                              Training dataset path.

optional arguments:
  -h, --help                        Show this help message and exit
  -o, --output [OUTPUT]             Output dir path.
  -m, --modes MODES [MODES ...]     Sampling mode. Default "all" stands for "fields", "mask" and "augment".
  -f, --fields FIELDS [FIELDS ...]  List of fields order. Default "all" stands for "tak" and "tka".
  -a, --augment [AUGMENT]           Model name for context augmentation mode.
```


**`train`**: Train one model for one sample.
```sh
python bagbert train experiments/pubmedbert-tak data/train-tak.csv data/val.csv


positional arguments:
  model                     Model path (Folder with config.json file).
  train                     Training dataset path.
  val                       Validation dataset path.

optional arguments:
  -h, --help                Show this help message and exit
  -f, --fields [FIELDS]     Selected fields order. Default "tak" for title-abstract-keywords.
  -c, --clean  [CLEAN]      Mask terms related to COVID-19. 0: False (default), 1: Remove, 2: Mask token.
  -e, --epochs [EPOCHS]     Maximum number of epochs if not stopped. Default 1000.
```


**`select`**: Select k-sub-models based on Hamming loss.
```sh
python bagbert select experiments


positional arguments:
  models                Experiments directory path.

optional arguments:
  -h, --help            Show this help message and exit
  -m, --min [MIN]       Minimum k sub-model per model. Default 1.
  -M, --max [MAX]       Maximum k sub-model per model. Default 5.
```


**`predict`**: Predict by average of inferences.
```sh
python bagbert predict experiments data/test.csv


positional arguments:
  models                Experiments directory path.
  path                  Dataset path.

optional arguments:
  -h, --help            Show this help message and exit.
  -o, --output [OUT]    Output pickle filename. Default "predictions.pkl".
```

## Weights
Due to the high size of one sub-model (~450MB),
we cannot provide all trained sub-models (~21GB). However, the initial weights of each model are available on the HF Hub. The model classes in `model.py` inherit the model methods from the transformers module. Initial weights are:
- [PubmedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) from [Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing](https://doi.org/10.1145/3458754) (Yu Gu et al.)
- [Covid-SciBERT](https://huggingface.co/lordtt13/COVID-SciBERT) (pretrained by Tanmay Thakur)
- [Clinical BERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/tree/main) from [Publicly Available Clinical BERT Embeddings](https://doi.org/10.18653/v1/W19-1909) (Alsentzer Emily et al.)
- [BioMed](https://huggingface.co/allenai/biomed_roberta_base) RoBERTa from [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://doi.org/10.18653/v1/2020.acl-main.740) (Gururangan Suchin et al.)

To load the initial weights trained with PyTorch to TF, use the `from_pt` argument. Save the model to TF mode.
```python
from bagbert.model import BERTTopic

model = BERTTopic.from_pretrained('pytorch/model/path', from_pt = True)
model.save_pretrained('experiments/model_name')
```

## Citation
If you find BagBERT useful in your research, please cite the following paper:
```bibtex
@misc{rakotoson2021bagbert,
  title={BagBERT: BERT-based bagging-stacking for multi-topic classification}, 
  author={Loïc Rakotoson and Charles Letaillieur and Sylvain Massip and Fréjus Laleye},
  year={2021},
  eprint={2111.05808},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```