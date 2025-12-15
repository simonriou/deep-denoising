import torch

"""
This function pads variable-length spectrograms in a batch to the maximum length in that batch.
It ensures that all tensors in the batch have the same time dimension by zero-padding shorter ones.
This is used for batching in the data loader when dealing with audio data of varying lengths.
"""

def pad_collate(batch):
    # Batch is list of tuples (feature, label)
    # feature shape: [1, Freq, Time]
    
    # Find max time length in this batch
    max_len = max([x[0].shape[2] for x in batch])
    freq_bins = batch[0][0].shape[1]
    
    # Initialize batch tensors
    batch_size = len(batch)
    features_padded = torch.zeros(batch_size, 1, freq_bins, max_len)
    labels_padded = torch.zeros(batch_size, 1, freq_bins, max_len)
    
    for i, (feat, lbl) in enumerate(batch):
        curr_len = feat.shape[2]
        features_padded[i, :, :, :curr_len] = feat
        labels_padded[i, :, :, :curr_len] = lbl
        
    return features_padded, labels_padded