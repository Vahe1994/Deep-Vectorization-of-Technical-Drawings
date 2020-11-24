from util_files.data.syndata.datasets import SyntheticHandcraftedDataset
from util_files.data.preprocessing import PreprocessedBase


class PreprocessedSyntheticHandcrafted(SyntheticHandcraftedDataset, PreprocessedBase):
    def __init__(self, **kwargs):
        PreprocessedBase.__init__(self, **kwargs)
        SyntheticHandcraftedDataset.__init__(self, **kwargs)
        
    def __getitem__(self, idx):
        return self.preprocess_sample(SyntheticHandcraftedDataset.__getitem__(self, idx))
