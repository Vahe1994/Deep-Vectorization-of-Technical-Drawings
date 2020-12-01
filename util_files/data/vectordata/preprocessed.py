from util_files.data.vectordata.prepatched import PrepatchedSVG
from util_files.data.preprocessing import PreprocessedBase, PreprocessedPacked


class PreprocessedSVG(PrepatchedSVG, PreprocessedBase):
    def __init__(self, **kwargs):
        PreprocessedBase.__init__(self, **kwargs)
        PrepatchedSVG.__init__(self, **kwargs)

    def __getitem__(self, idx):
        return self.preprocess_sample(PrepatchedSVG.__getitem__(self, idx))


class PreprocessedSVGPacked(PrepatchedSVG, PreprocessedPacked):
    def __init__(self, **kwargs):
        PreprocessedPacked.__init__(self, **kwargs)
        PrepatchedSVG.__init__(self, **kwargs)

    def __getitem__(self, idx):
        return self.preprocess_sample(PrepatchedSVG.__getitem__(self, idx))
