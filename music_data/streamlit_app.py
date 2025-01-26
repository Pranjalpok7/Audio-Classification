import streamlit as st
import torch
import torch.nn.functional as F
import os.path as osp
import librosa
import numpy as np
import resampy
from torch_geometric.nn import GCNConv, GATConv, knn_graph
import pandas as pd
import tensorflow_hub as hub

# Configuration
DEVICE = torch.device('cpu') # Force CPU usage
K = 15

# Load models and configuration once
@st.cache_resource
def load_model_and_data():
    # Best hyperparameters for GAT
    GAT_HYPERPARAMS = {
    'hidden_dim': 24,
    'dropout': 0.341485248071941,
    'lr': 0.0028413205322696086,
    'weight_decay': 2.851478550278825e-05,
    'heads': 4
    }

    # Best hyperparameters for GCN
    GCN_HYPERPARAMS = {
        'hidden_dim': 48,
        'dropout': 0.35383540022461457,
        'lr': 0.0059797792015148545,
        'weight_decay': 3.398807978977328e-05
    }
    
    # Model Definitions
    class GCN(torch.nn.Module):
        def __init__(self, num_features, num_classes):
            super().__init__()
            self.conv1 = GCNConv(num_features, GCN_HYPERPARAMS['hidden_dim'])
            self.bn1 = torch.nn.BatchNorm1d(GCN_HYPERPARAMS['hidden_dim'])
            self.conv2 = GCNConv(GCN_HYPERPARAMS['hidden_dim'], num_classes)
            self.dropout = GCN_HYPERPARAMS['dropout']

        def forward(self, x, edge_index):
            x = F.relu(self.bn1(self.conv1(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    class GAT(torch.nn.Module):
        def __init__(self, num_features, num_classes):
            super().__init__()
            self.conv1 = GATConv(
                num_features,
                GAT_HYPERPARAMS['hidden_dim'],
                heads=GAT_HYPERPARAMS['heads']
            )
            self.bn1 = torch.nn.BatchNorm1d(GAT_HYPERPARAMS['hidden_dim'] * GAT_HYPERPARAMS['heads'])
            self.conv2 = GATConv(
                GAT_HYPERPARAMS['hidden_dim'] * GAT_HYPERPARAMS['heads'],
                num_classes,
                heads=1,
                concat=False
            )
            self.dropout = GAT_HYPERPARAMS['dropout']

        def forward(self, x, edge_index):
            x = F.relu(self.bn1(self.conv1(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    # Load GAT model
    try:
        gat_model = GAT(1024, 10).to(DEVICE).eval()
        gat_model.load_state_dict(torch.load('best_GAT.pt', map_location=DEVICE))
    except Exception as e:
        st.error(f"Error loading GAT model: {e}")
        return None, None, None, None, None, None
    
    # Load GCN model
    try:
        gcn_model = GCN(1024, 10).to(DEVICE).eval()
        gcn_model.load_state_dict(torch.load('best_GCN.pt', map_location=DEVICE))
    except Exception as e:
        st.error(f"Error loading GCN model: {e}")
        return None, None, None, None, None, None

    # Load YAMNet (if needed)
    try:
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    except Exception as e:
         st.error(f"Error loading yamnet_model: {e}")
         return None, None, None, None, None, None
    
    # Load existing data and normalization parameters
    try:
        data = torch.load(osp.join('all_graph_data/processed', 'data.pt'), map_location=DEVICE)
    except Exception as e:
        st.error(f"Error loading graph data: {e}")
        return None, None, None, None, None, None
    
    try:
      norm_params = torch.load(osp.join('all_graph_data/processed', 'norm_params.pt'), map_location=DEVICE)
    except Exception as e:
         st.error(f"Error loading normalization params: {e}")
         return None, None, None, None, None, None
    
    # Load metadata and create mapping
    try:
      metadata = pd.read_csv('nepali_music_metadata.csv')
      class_mapping = dict(zip(metadata['class_id'], metadata['class_name']))
    except Exception as e:
         st.error(f"Error loading metadata: {e}")
         return None, None, None, None, None, None


    return gat_model, gcn_model, yamnet_model, data, norm_params, class_mapping
    
gat_model, gcn_model, yamnet_model, data, norm_params, class_mapping  = load_model_and_data()


def predict_new_audio(audio_path, model_type):
    """
    Predict the class of a new audio file using the trained GNN model.

    Args:
        audio_path (str): Path to the audio file (30s WAV format).
        model_type (str): 'gat' or 'gcn' for the model type.

    Returns:
        str: Predicted class name.
    """
    try:
        # Step 1: Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        if sr != 16000:  # Resample to YAMNet's expected sample rate
            audio = resampy.resample(audio, sr, 16000)
            
        # Step 2: Extract YAMNet features
        _, yamnet_embeddings, _ = yamnet_model(audio)
        new_feature = torch.tensor(np.mean(yamnet_embeddings.numpy(), axis=0), dtype=torch.float32)

        # Step 3: Normalize features using training data statistics
        new_feature = (new_feature - norm_params['min']) / (norm_params['max'] - norm_params['min'] + 1e-8)
        
        # Check if new_feature has NaN or Inf
        if torch.isnan(new_feature).any() or torch.isinf(new_feature).any():
             st.error(f"Error: Normalized feature has NaN or Inf values for {audio_path}")
             return None
        
        # Step 4: Move all tensors to the same device
        data.x = data.x.to(DEVICE)  # Move graph features to the same device as the model
        new_feature = new_feature.to(DEVICE)  # Ensure new feature is on the same device

        # Step 5: Create extended graph
        combined_features = torch.cat([data.x, new_feature.unsqueeze(0)])
        edge_index = knn_graph(combined_features, k=K).to(DEVICE)


        # Step 6: Select model and make prediction
        if model_type == 'gat':
          model = gat_model
        else:
          model = gcn_model
        with torch.no_grad():
            out = model(combined_features, edge_index)
            predicted_class_index = out[-1].argmax().item()

        # Step 7: Map class index to class name
        predicted_class_name = class_mapping.get(predicted_class_index, "Unknown")
        return predicted_class_name
    except Exception as e:
        st.error(f"Error processing {audio_path}: {str(e)}")
        return None  # Return None if inference fails

# Streamlit App
st.title('Nepali Music Genre Classification')
st.markdown("This app classifies audio samples into different Nepali music genres using a Graph Neural Network.")
st.markdown("Please upload a WAV file with a maximum length of 30 seconds")
# File Upload
uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

# Model Type Selection
model_type = st.selectbox("Select GNN Model Type", ["gat", "gcn"])

if uploaded_file is not None:
    # Save audio to a temp file
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    st.audio("temp_audio.wav")

    # Perform prediction
    predicted_class = predict_new_audio("temp_audio.wav", model_type)
    
    # Display results
    if predicted_class:
        st.success(f"Predicted genre: **{predicted_class}**")
    else:
      st.error("Failed to process this audio file.")
else:
    st.info("Please upload an audio file to classify.")