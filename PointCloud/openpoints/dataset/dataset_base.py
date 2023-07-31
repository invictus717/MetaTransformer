
import torch
from torch.utils.data import dataset
from tqdm import tqdm
from pathlib import Path


class DatasetBase(dataset.Dataset):
    def __init__(self, 
                 dataset_name, split,
                 cache_dir = None, 
                 load_cache_if_exists=True, 
                 **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir
        
        self.is_cached = False
        if load_cache_if_exists:
            self.cache(verbose=0, must_exist=True)
        
    @property
    def record_tokens(self):
        raise NotImplementedError
    
    def read_record(self, token):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.record_tokens)
    
    def __getitem__(self, index):
        token  = self.record_tokens[index]
        try:
            return self._records[token]
        except AttributeError:
            record = self.read_record(token)
            self._records = {token:record}
            return record
        except KeyError:
            record = self.read_record(token)
            self._records[token] = record
            return record
    
    def read_all_records(self, verbose=1):
        self._records = {}
        if verbose:
            print(f'Reading all {self.split} records...', flush=True)
            for token in tqdm(self.record_tokens):
                self._records[token] = self.read_record(token)
        else:
            for token in self.record_tokens:
                self._records[token] = self.read_record(token)
    
    def get_cache_path(self, path=None):
        if path is None: path = self.cache_dir
        base_path = (Path(path)/self.dataset_name)/self.split
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path
    
    def cache_load_and_save(self, base_path, op, verbose):
        tokens_path = base_path/'tokens.pt'
        records_path = base_path/'records.pt'
        
        if op == 'load':
            self._record_tokens = torch.load(str(tokens_path))
            self._records = torch.load(str(records_path))
        elif op == 'save':
            if tokens_path.exists() and records_path.exists() \
                and hasattr(self, '_record_tokens') and hasattr(self, '_records'):
                return
            self.read_all_records(verbose=verbose)
            torch.save(self.record_tokens, str(tokens_path))
            torch.save(self._records, str(records_path))
        else:
            raise ValueError(f'Unknown operation: {op}')
    
    def cache(self, path=None, verbose=1, must_exist=False):
        if self.is_cached: return
        
        base_path = self.get_cache_path(path)
        try:
            if verbose: print(f'Trying to load {self.split} cache from disk...', flush=True)
            self.cache_load_and_save(base_path, 'load', verbose)
            if verbose: print(f'Loaded {self.split} cache from disk.', flush=True)
        except FileNotFoundError:
            if must_exist: return
            
            if verbose: print(f'{self.split} cache does not exist! Cacheing...', flush=True)
            self.cache_load_and_save(base_path, 'save', verbose)
            if verbose: print(f'Saved {self.split} cache to disk.', flush=True)
        
        self.is_cached = True


