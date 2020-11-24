import numpy as np
from torch.multiprocessing import Process
import torch

from util_files.data.preprocessed import PreprocessedDataset


class ChunkedDatasetLoader:
    def __init__(self, dataset, chunk_size=None, memory_constraint=None, batch_size=1, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        input, target = dataset[0]
        input_size = input.element_size() * input.numel()
        target_size = target.element_size() * target.numel()
        
        # memory_constraint should be diveded by 2 since there is next_chunk and current_chunk
        self.chunk_size = get_chunk_size(input_size + target_size, chunk_size, memory_constraint / 2, batch_size)
        
        self.data = ChunkedDatasetIter(dataset, self.chunk_size)
        self.size = self.data.get_batches_n(batch_size=batch_size)
        
        self.initialized = False
    
        
    def __len__(self):
        return self.size
    
    
    def __iter__(self):
        if not self.initialized: self.initialize()
        
        self.data.reset_head()
        self.prepare_next_chunk()
        while self.data.next_size() > 0:
            self.next_chunk_loader.join()
            inputs = self.input_buffers[self.next_buffer_i][:self.data.next_size()]
            targets = self.target_buffers[self.next_buffer_i][:self.data.next_size()]
            
            self.data.move_head()
            self.next_buffer_i = 1 - self.next_buffer_i
            self.prepare_next_chunk()
            
            if self.shuffle: ids = torch.randperm(len(inputs))
            else: ids = torch.arange(0, len(inputs))
            for b in ((inputs[ids[start:start+self.batch_size]], targets[ids[start:start+self.batch_size]]) for start in torch.arange(0, len(ids), self.batch_size)):
                yield b
        raise StopIteration
       
    
    def initialize(self):
        self.allocate_buffers()
        self.next_buffer_i = 0
        self.initialized = True
    
    
    def allocate_buffers(self):
        input, target = self.data.dataset[0]
        input_shape = input.shape
        target_shape = target.shape
        input_type = input.dtype
        target_type = target.dtype
        max_chunk_size = max(self.data.all_chunk_sizes())
    
        self.input_buffers = [torch.empty(max_chunk_size, *input_shape, dtype=input_type).share_memory_() for _ in (1,2)]
        self.target_buffers = [torch.empty(max_chunk_size, *target_shape, dtype=target_type).share_memory_() for _ in (1,2)]       

        
    def prepare_next_chunk(self):        
        self.next_chunk_loader = Process(target=self.fill_next_chunk)
        self.next_chunk_loader.start()
    

    def fill_next_chunk(self):
        torch.set_num_threads(1)
        self.data.write_next_to(self.input_buffers[self.next_buffer_i], self.target_buffers[self.next_buffer_i])


class ChunkedConcatDatasetLoader(ChunkedDatasetLoader):
    def __init__(self, datasets, **kwargs):
        super().__init__(datasets[0], **kwargs)
        self.data = ConcatChunkedDatasetIter(datasets, self.chunk_size)
        self.size = self.data.get_batches_n(batch_size=kwargs['batch_size'])


class ChunkedDatasetIter:
    def __init__(self, dataset:PreprocessedDataset, chunk_size):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.chunk_i = 0
        self.head = 0


    def write_next_to(self, inputs, targets):
        size = self.next_size()
        inputs[:size] = self.dataset.images[self.head : self.head + size]
        targets[:size] = self.dataset.targets[self.head : self.head + size]

    
    def reset_head(self):
        self.chunk_i = 0
        self.head = 0
        
        
    def move_head(self):
        self.chunk_i += 1
        self.head = np.round(self.chunk_size * self.chunk_i).astype(int)
    
    
    def next_size(self):
        return min(len(self.dataset), np.round(self.chunk_size * (self.chunk_i+1)).astype(int)) - self.head
    
    
    def all_chunk_sizes(self):
        heads = np.arange(0, len(self.dataset), self.chunk_size).round().astype(int)
        sizes = np.append(heads[1:] - heads[:-1], [len(self.dataset) - heads[-1]])
        return sizes

    
    def get_batches_n(self, batch_size):
        return np.ceil(self.all_chunk_sizes() / batch_size).astype(int).sum()


class ConcatChunkedDatasetIter:
    def __init__(self, datasets, chunk_size):        
        dataset_probas = np.array([len(dataset) for dataset in datasets])
        dataset_probas = dataset_probas / dataset_probas.sum()
        chunk_sizes = dataset_probas * chunk_size
        self.iters = [ChunkedDatasetIter(dataset, cs) for dataset,cs in zip(datasets, chunk_sizes)]
        self.dataset = self.iters[0].dataset


    def write_next_to(self, inputs, targets):
        start = 0
        for it in self.iters:
            end = start + it.next_size()
            it.write_next_to(inputs[start:end], targets[start:end])
            start = end
        
    
    def reset_head(self):
        for it in self.iters:
            it.reset_head()
       
    
    def move_head(self):
        for it in self.iters:
            it.move_head()
            
    
    def next_size(self):
        return sum(it.next_size() for it in self.iters)
    
    
    def all_chunk_sizes(self):
        sizes = [it.all_chunk_sizes() for it in self.iters]
        max_chunks_n = max(len(s) for s in sizes)
        return np.sum([np.pad(s, (0, max_chunks_n - len(s)), 'constant', constant_values=0) for s in sizes], axis=0)
    
    
    def get_batches_n(self, batch_size):
        return np.ceil(self.all_chunk_sizes() / batch_size).astype(int).sum()


def get_chunk_size(sample_size, chunk_size=None, memory_constraint=None, batch_size=1):
    if memory_constraint is None:
        assert chunk_size is not None, 'Either memory_constraint or chunk_size are required'
    else:
        constrained_chunk_size = memory_constraint / sample_size
        constrained_chunk_size = int(constrained_chunk_size // batch_size) * batch_size
        if chunk_size is None or chunk_size > constrained_chunk_size:
            chunk_size = constrained_chunk_size
    return chunk_size
