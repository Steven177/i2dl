"""Definition of Dataloader"""

import numpy as np
import math


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in the notebook.                                                 #
        
        ########################################################################
        
        
        ########################################################################        
       
        # Initialize variables
        batch = []    
        batches_a = []
        
        # Create a iterator
        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(self.dataset)))
        else:
            index_iterator = iter(range(len(self.dataset)))
        
        
        # Step 1
        for index in index_iterator:
            data_dict = self.dataset[index]
            batch.append(data_dict)
            
            if len(batch) == self.batch_size:
                batches_a.append(batch)
                batch = []
        batches_a.append(batch)
        
        # Remove last batch if drop_last
        if self.drop_last:
            if len(batches_a[-1]) < self.batch_size:
                del batches_a[-1]
                
        # Step 2
        batches_dic = []
        for batch in batches_a:
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)
            batches_dic.append(batch_dict)
  
        # Step 3
        numpy_batches = []
        for batch in batches_dic:
            numpy_batch = {}
            for key, value in batch.items():
                numpy_batch[key] = np.array(value)
            numpy_batches.append(numpy_batch)
            
        # Create iterator
        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(numpy_batches)))
        else:
            index_iterator = iter(range(len(numpy_batches)))   
           
        # Yield data
        for index in index_iterator:
            yield(numpy_batches[index])
                             
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last!                                 #
        ########################################################################

        # length = round(len(self.dataset) / self.batch_size)
        # if self.drop_last:
            # length -= 1
        
        length = math.floor(len(self.dataset) / self.batch_size)
        rest = len(self.dataset) % self.batch_size
        if self.drop_last == False and rest != 0:
            length += 1

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
