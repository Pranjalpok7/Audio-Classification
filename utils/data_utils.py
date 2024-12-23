import torchaudio
import torch
import numpy as np
def extract_features(audio_path, model,model_type,device):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)
    
    if model_type == 'vggish':
        if waveform.shape[0]>1:
            waveform= torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if required
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        #Extract VGGish features
        with torch.no_grad():
            embeddings = model(waveform)
            avg_embeddings = torch.mean(embeddings, dim=0)
            return avg_embeddings.cpu().numpy()
    elif model_type == 'yamnet':
       
       if waveform.shape[0]>1:
          waveform= torch.mean(waveform, dim=0, keepdim=True)
         
       if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
       
       segment_length = 0.96
       hop_length = 0.48
       total_samples = waveform.shape[1]
       segment_samples = int(sample_rate * segment_length)
       hop_samples = int(sample_rate * hop_length)
       
       all_embeddings = []
       
       for i in range(0,total_samples - segment_samples + 1 , hop_samples):
             segment = waveform[:, i : i + segment_samples]
             with torch.no_grad():
                output = model(segment)
                embeddings = output.logits.mean(dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
       all_embeddings = np.array(all_embeddings)
       avg_embeddings = all_embeddings.mean(axis=0)
       return avg_embeddings
    elif model_type == 'panns':
            
            if waveform.shape[0]>1:
                waveform= torch.mean(waveform, dim=0, keepdim=True)
            
            if sample_rate != 32000:
                resampler = torchaudio.transforms.Resample(sample_rate, 32000)
                waveform = resampler(waveform)
            
            with torch.no_grad():
                output = model(waveform.cpu())
                avg_embeddings = output['embedding'].numpy().squeeze()
                return avg_embeddings