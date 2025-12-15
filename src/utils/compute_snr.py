import torch

def compute_snr(ref, proc):
    """
    Compute SNR in dB between reference and processed signals.
    
    ref: numpy array of shape (N,)
    proc: same shape as ref
    """
    # Convert to torch tensor if needed
    if not isinstance(ref, torch.Tensor):
        ref = torch.tensor(ref, dtype=torch.float32)
    if not isinstance(proc, torch.Tensor):
        proc = torch.tensor(proc, dtype=torch.float32)
        
    noise = proc - ref
    power_signal = torch.mean(ref ** 2)
    power_noise = torch.mean(noise ** 2)
    
    snr_db = 10 * torch.log10(power_signal / (power_noise + 1e-8))
    return snr_db.item()