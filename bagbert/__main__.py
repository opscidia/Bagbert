import argparse, sys, re, pickle
from pprint import pprint as cat

from utils import init_env
from dataset import Augmenter, create_samples
from trainer import train, weighted_select
from model import MetaEnsemble


class BagBert:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            prog = 'bagbert',
            description = 'BagBert commands',
            usage = """python bagbert <command> [<args>]

The expected commands are:
   sample     Create CSV samples from training dataset.
   train      Train one model for one sample.
   select     Select sub-models based on Hamming loss.
   predict    Predict by average of inferences.
""")
        parser.add_argument('command', help = 'Command to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        getattr(self, args.command)()
    
    def sample(self):
        parser = argparse.ArgumentParser(
            description = 'Create CSV samples from training dataset.')
        parser.add_argument('path', help = 'Training dataset path.')
        parser.add_argument(
            '-o', '--output', nargs = '?',
            help = 'Output dir path.')
        parser.add_argument(
            '-m', '--modes', nargs = '+',
            default = ['all'],
            help = 'Sampling mode. Default "all" stands for "fields", "mask" and "augment".')
        parser.add_argument(
            '-f', '--fields', nargs = '+',
            default = ['all'],
            help = 'List of fields order. Default "all" stands for "tak" and "tka".')
        parser.add_argument(
            '-a', '--augment', nargs = '?',
            help = 'Model name for context augmentation mode.')
        args = parser.parse_args(sys.argv[2:])
        
        _modes = ['fields', 'mask', 'augment', 'all']
        modes = args.modes
        for m in modes:
            if m not in _modes: raise ValueError(f"Unrecognized {m} mode.")
            if len(modes) == 1: modes = modes[0]
        fields = args.fields
        
        for f in fields:
            if not re.search(r'[akt]', f): raise ValueError(f"Unrecognized {f} field order.")
        if len(fields) == 1: fields = fields[0]

        output = args.output if args.output else '.'
        augmenter = Augmenter(args.augment)

        create_samples(
            args.path, output,
            modes, fields, augmenter
        )
    


    def train(self):
        parser = argparse.ArgumentParser(
            description = 'Train one model for one sample.')
        parser.add_argument('model', help = 'Model path (Folder with config.json file).')
        parser.add_argument('train', help = 'Training dataset path.')
        parser.add_argument('val', help = 'Validation dataset path.')
        parser.add_argument(
            '-f', '--fields', default = 'tak', nargs = '?',
            help = 'Selected fields order. Default "tak" for title-abstract-keywords')
        parser.add_argument(
            '-c', '--clean', nargs = '?',
            default = 0, type = int,
            help = 'Mask terms related to COVID-19. 0: False (default), 1: Remove, 2: Mask token.')
        parser.add_argument(
            '-e', '--epochs', nargs = '?',
            default = 1e3, type = int,
            help = 'Maximum number of epochs if not stopped. Default 1000')
        args = parser.parse_args(sys.argv[2:])

        clean = bool(args.clean) if args.clean != 2 else 'mask'
        strategy = init_env()

        train(
            args.model, args.train, args.val,
            args.fields, clean, strategy
        )
        
        

    def select(self):
        parser = argparse.ArgumentParser(
            description = 'Select sub-models based on Hamming loss.')
        parser.add_argument('models', help = 'Experiments directory path.')
        parser.add_argument(
            '-m', '--min', nargs = '?',
            default = 1, type = int,
            help = 'Minimum k sub-model per model.')
        parser.add_argument(
            '-M', '--max', nargs = '?',
            default = 5, type = int,
            help = 'Maximum k sub-model per model.')
        args = parser.parse_args(sys.argv[2:])

        min_k, max_k = args.min, args.max
        assert min_k >= 0, "Negative k is meaningless."
        assert min_k < max_k, "The minimum cannot be the maximum."

        weighted_select(
            args.models,
            min_k, max_k
        )
    


    def predict(self):
        parser = argparse.ArgumentParser(
            description = 'Predict by average of inferences.')
        parser.add_argument('models', help = 'Experiments directory path.')
        parser.add_argument('path', help = 'Dataset path.')
        parser.add_argument(
            '-o', '--output', nargs = '?',
            default = 'predictions.pkl',
            help = 'Output pickle filename. Default "predictions.pkl".')

        args = parser.parse_args(sys.argv[2:])

        model = MetaEnsemble.from_pretrained(args.models)
        pred = model.predict(args.path)
        with open(args.output, 'wb') as f:
            pickle.dump(pred, f)
        



BagBert()