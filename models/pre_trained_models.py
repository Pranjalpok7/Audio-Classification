#Load the VGGishModel
class VGGishModel(torch.nn.Module):
    def __init__(self):
        super(VGGishModel, self).__init__()
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')

    def forward(self, audio):
        embeddings = self.model(audio)
        return embeddings


#Load the YAMNet Model 
'''
import torch
import torchaudio
class YAMNetModel(torch.nn.Module):
    def __init__(self):
        super(YAMNetModel, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_large', pretrained=True)
        
    def forward(self,audio):
            
             
        return self.model
'''
#Load the PANNs Model
import torch
from panns_inference import AudioTagging
class PANNsModel(torch.nn.Module):
    def __init__(self):
       super(PANNsModel, self).__init__()
       self.model = AudioTagging(checkpoint_path='./pretrained_models/PANNs/Wavegram_Logmel_B1.pth')

    def forward(self,audio):
         
          return self.model(audio)