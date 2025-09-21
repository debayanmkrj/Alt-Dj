import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict

class AudioStyleTransferDataset(Dataset):
    """Dataset for training audio style transfer between DJ styles"""
    def __init__(self, audio_features, style_labels, target_audio_features):
        self.audio_features = torch.FloatTensor(audio_features)
        self.style_labels = torch.LongTensor(style_labels)
        self.target_audio_features = torch.FloatTensor(target_audio_features)
    
    def __len__(self):
        return len(self.audio_features)
    
    def __getitem__(self, idx):
        return self.audio_features[idx], self.style_labels[idx], self.target_audio_features[idx]

class AudioStyleTransferNetwork(nn.Module):
    """Neural Network for DJ Audio Style Transfer"""
    def __init__(self, input_size=128, style_embedding_size=64, hidden_sizes=[256, 512, 256]):
        super(AudioStyleTransferNetwork, self).__init__()
        
        # Style embedding layer
        self.style_embedding = nn.Embedding(5, style_embedding_size)  # 5 DJ styles
        
        # Audio encoder
        encoder_layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            current_size = hidden_size
        
        self.audio_encoder = nn.Sequential(*encoder_layers)
        
        # Style-conditioned decoder
        decoder_input_size = current_size + style_embedding_size
        decoder_layers = []
        
        for hidden_size in reversed(hidden_sizes[:-1]):
            decoder_layers.extend([
                nn.Linear(decoder_input_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            decoder_input_size = hidden_size
        
        # Output layer
        decoder_layers.append(nn.Linear(decoder_input_size, input_size))
        decoder_layers.append(nn.Tanh())  # Bounded output
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, audio_features, style_id):
        # Encode audio
        encoded_audio = self.audio_encoder(audio_features)
        
        # Get style embedding
        style_embed = self.style_embedding(style_id)
        
        # Concatenate encoded audio with style embedding
        combined = torch.cat([encoded_audio, style_embed], dim=1)
        
        # Decode to styled audio features
        styled_features = self.decoder(combined)
        
        return styled_features

class AudioStyleTransferTrainer:
    def __init__(self, 
                 audio_folder="audio_training",
                 trained_results_path="altdj_trained_results.json",
                 output_model_path="audio_style_transfer_model.pt"):
        
        self.audio_folder = audio_folder
        self.trained_results_path = trained_results_path
        self.output_model_path = output_model_path
        
        # DJ mappings
        self.dj_names = ['armin', 'calvin', 'marshmello', 'parra', 'steve']
        self.dj_to_id = {name: i for i, name in enumerate(self.dj_names)}
        self.id_to_dj = {i: name for i, name in enumerate(self.dj_names)}
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        
        # Load interaction predictions
        self.load_interaction_predictions()
    
    def load_interaction_predictions(self):
        """Load existing interaction predictions from your trained model"""
        print("Loading interaction predictions...")
        
        if os.path.exists(self.trained_results_path):
            with open(self.trained_results_path, 'r') as f:
                results = json.load(f)
            
            self.dj_predictions = results.get('dj_ml_predictions', {})
            print(f"Loaded predictions for {len(self.dj_predictions)} DJs")
        else:
            print("No interaction predictions found - will train without them")
            self.dj_predictions = {}
    
    def extract_comprehensive_audio_features(self, audio_path, duration=120):
        """Extract comprehensive audio features for style transfer training"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, duration=duration)
            
            if len(y) == 0:
                return None
            
            # Spectral features
            hop_length = 512
            n_fft = 2048
            
            # Short-time Fourier transform
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            
            # Mel-frequency features
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=hop_length)
            log_mel = librosa.power_to_db(mel_spec)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
            
            # Chroma and tempo features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # RMS energy
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            
            # Combine features into comprehensive representation
            # We'll use statistics (mean, std, min, max) of temporal features
            feature_vector = []
            
            # Spectral statistics
            feature_vector.extend([
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                np.mean(zero_crossing_rate), np.std(zero_crossing_rate),
                np.mean(rms), np.std(rms)
            ])
            
            # MFCC statistics
            for i in range(min(13, mfccs.shape[0])):
                if i < mfccs.shape[0] and mfccs[i].size > 0:
                    feature_vector.extend([
                        np.mean(mfccs[i]), np.std(mfccs[i])
                    ])
                else:
                    feature_vector.extend([0.0, 0.0])
            
            # Chroma statistics
            for i in range(min(12, chroma.shape[0])):
                if i < chroma.shape[0] and chroma[i].size > 0:
                    feature_vector.extend([
                        np.mean(chroma[i]), np.std(chroma[i])
                    ])
                else:
                    feature_vector.extend([0.0, 0.0])
            
            # Mel spectrogram statistics (reduced dimensionality)
            mel_stats = []
            for i in range(0, min(64, log_mel.shape[0]), 4):  # Sample every 4th mel band
                if i < log_mel.shape[0] and log_mel[i].size > 0:
                    mel_stats.extend([
                        np.mean(log_mel[i]), np.std(log_mel[i])
                    ])
                else:
                    mel_stats.extend([0.0, 0.0])
            feature_vector.extend(mel_stats)
            
            # Tempo and rhythm
            feature_vector.append(float(tempo))
            
            # Ensure all elements are scalars and finite
            feature_vector = [float(x) if np.isfinite(x) else 0.0 for x in feature_vector]
            
            # Ensure consistent feature vector length
            target_length = 128
            if len(feature_vector) > target_length:
                feature_vector = feature_vector[:target_length]
            elif len(feature_vector) < target_length:
                feature_vector.extend([0.0] * (target_length - len(feature_vector)))
            
            return np.array(feature_vector, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def create_style_transfer_training_data(self):
        """Create training data for audio style transfer"""
        print("Creating style transfer training data...")
        
        all_features = []
        all_style_labels = []
        all_target_features = []
        
        # Extract features for each DJ
        dj_features = {}
        for dj_name in self.dj_names:
            audio_file = f"{dj_name}.mp3"
            audio_path = os.path.join(self.audio_folder, audio_file)
            
            if os.path.exists(audio_path):
                print(f"Processing {dj_name}...")
                
                # Extract features from multiple segments for data augmentation
                segments_features = []
                
                # Load full track
                y_full, sr = librosa.load(audio_path, duration=300)  # 5 minutes max
                
                # Extract multiple overlapping segments
                segment_length = 30  # 30 seconds per segment
                hop_length_seg = 15   # 15 seconds hop
                
                for start_time in np.arange(0, len(y_full)/sr - segment_length, hop_length_seg):
                    end_time = start_time + segment_length
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    
                    segment = y_full[start_sample:end_sample]
                    
                    # Save segment temporarily
                    temp_path = f"temp_segment_{dj_name}.wav"
                    sf.write(temp_path, segment, sr)
                    
                    # Extract features
                    features = self.extract_comprehensive_audio_features(temp_path, duration=segment_length)
                    
                    if features is not None:
                        segments_features.append(features)
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                dj_features[dj_name] = segments_features
                print(f"Extracted {len(segments_features)} feature segments for {dj_name}")
        
        # Create cross-style training pairs
        for source_dj in self.dj_names:
            for target_dj in self.dj_names:
                if source_dj != target_dj and source_dj in dj_features and target_dj in dj_features:
                    
                    source_features = dj_features[source_dj]
                    target_features = dj_features[target_dj]
                    
                    # Create training pairs (source->target style transfer)
                    min_segments = min(len(source_features), len(target_features))
                    
                    for i in range(min_segments):
                        all_features.append(source_features[i])
                        all_style_labels.append(self.dj_to_id[target_dj])  # Target style
                        all_target_features.append(target_features[i])     # Target features
        
        print(f"Created {len(all_features)} training samples")
        
        if len(all_features) == 0:
            print("No training data created!")
            return None, None, None
        
        return np.array(all_features), np.array(all_style_labels), np.array(all_target_features)
    
    def train_style_transfer_network(self, epochs=150, batch_size=16, learning_rate=0.001):
        """Train the audio style transfer neural network"""
        print("Training Audio Style Transfer Neural Network...")
        
        # Create training data
        X, y_style, X_target = self.create_style_transfer_training_data()
        
        if X is None or len(X) == 0:
            print("Failed to create training data - checking audio files...")
            
            # Check if audio files exist and are readable
            for dj_name in self.dj_names:
                audio_file = f"{dj_name}.mp3"
                audio_path = os.path.join(self.audio_folder, audio_file)
                
                if os.path.exists(audio_path):
                    try:
                        y_test, sr = librosa.load(audio_path, duration=10)  # Test load
                        print(f"{dj_name}: OK (duration: {len(y_test)/sr:.1f}s)")
                    except Exception as e:
                        print(f"{dj_name}: ERROR - {e}")
                else:
                    print(f"{dj_name}: FILE NOT FOUND")
            
            print("Unable to train without valid audio data")
            return None
        
        print(f"Training data created: {len(X)} samples")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_target_scaled = self.scaler.transform(X_target)
        
        # Split data
        X_train, X_test, y_style_train, y_style_test, X_target_train, X_target_test = train_test_split(
            X_scaled, y_style, X_target_scaled, test_size=0.2, random_state=42, stratify=y_style
        )
        
        # Create datasets
        train_dataset = AudioStyleTransferDataset(X_train, y_style_train, X_target_train)
        test_dataset = AudioStyleTransferDataset(X_test, y_style_test, X_target_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = X_scaled.shape[1]
        self.model = AudioStyleTransferNetwork(input_size=input_size)
        
        # Loss functions
        mse_loss = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0
            
            for batch_features, batch_styles, batch_targets in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                styled_features = self.model(batch_features, batch_styles)
                
                # Calculate loss (how close are styled features to target features)
                loss = mse_loss(styled_features, batch_targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Testing
            self.model.eval()
            epoch_test_loss = 0
            
            with torch.no_grad():
                for batch_features, batch_styles, batch_targets in test_loader:
                    styled_features = self.model(batch_features, batch_styles)
                    loss = mse_loss(styled_features, batch_targets)
                    epoch_test_loss += loss.item()
            
            avg_test_loss = epoch_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_test_loss)
            
            # Save best model
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                self.save_model()
            
            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        
        print(f"Training completed! Best test loss: {best_test_loss:.6f}")
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Audio Style Transfer Training')
        plt.savefig('style_transfer_training.png')
        plt.show()
        
        return train_losses, test_losses
    
    def save_model(self):
        """Save the trained model and scaler"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'dj_names': self.dj_names,
            'dj_to_id': self.dj_to_id,
            'id_to_dj': self.id_to_dj,
            'model_architecture': {
                'input_size': 128,
                'style_embedding_size': 64,
                'hidden_sizes': [256, 512, 256]
            }
        }, self.output_model_path)
        
        print(f"Model saved to {self.output_model_path}")
    
    def load_model(self):
        """Load trained model"""
        if not os.path.exists(self.output_model_path):
            print(f"Model file not found: {self.output_model_path}")
            return False
        
        checkpoint = torch.load(self.output_model_path, map_location='cpu')
        
        # Rebuild model
        self.model = AudioStyleTransferNetwork(
            input_size=checkpoint['model_architecture']['input_size'],
            style_embedding_size=checkpoint['model_architecture']['style_embedding_size'],
            hidden_sizes=checkpoint['model_architecture']['hidden_sizes']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.dj_names = checkpoint['dj_names']
        self.dj_to_id = checkpoint['dj_to_id']
        self.id_to_dj = checkpoint['id_to_dj']
        
        self.model.eval()
        print("Audio Style Transfer model loaded successfully")
        return True
    
    def apply_neural_style_transfer(self, source_audio_path, target_dj_style, output_path=None):
        """Apply neural network style transfer to audio"""
        if not self.model:
            print("No model loaded!")
            return None
        
        if target_dj_style not in self.dj_to_id:
            print(f"Unknown DJ style: {target_dj_style}")
            return None
        
        try:
            print(f"Applying {target_dj_style} neural style transfer...")
            
            # Extract features from source audio
            source_features = self.extract_comprehensive_audio_features(source_audio_path)
            
            if source_features is None:
                print("Failed to extract source features")
                return None
            
            # Scale features
            source_features_scaled = self.scaler.transform([source_features])
            
            # Convert to tensor
            source_tensor = torch.FloatTensor(source_features_scaled)
            target_style_id = torch.LongTensor([self.dj_to_id[target_dj_style]])
            
            # Apply style transfer
            with torch.no_grad():
                styled_features = self.model(source_tensor, target_style_id)
                styled_features_np = styled_features.cpu().numpy()[0]
            
            # Convert styled features back to audio (this is simplified)
            # In practice, you'd need a more sophisticated feature->audio conversion
            styled_audio = self.features_to_audio(styled_features_np, source_audio_path)
            
            # Save styled audio
            if output_path is None:
                output_path = f"neural_styled_{target_dj_style}_{int(time.time())}.wav"
            
            sf.write(output_path, styled_audio, 22050)
            print(f"Neural style transfer complete: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Error in neural style transfer: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def features_to_audio(self, styled_features, reference_audio_path):
        """Convert styled features back to audio (simplified approach)"""
        try:
            # Load reference audio
            y_ref, sr = librosa.load(reference_audio_path, duration=120)
            
            # This is a simplified approach - in practice you'd need more sophisticated methods
            # For now, we'll apply the style as filtering and effects on the original audio
            
            # Extract key style parameters from features
            spectral_centroid_style = styled_features[0]
            spectral_rolloff_style = styled_features[2]
            energy_style = styled_features[8]
            
            # Apply basic style transfer effects
            styled_audio = y_ref.copy()
            
            # Simple EQ based on styled features
            if spectral_centroid_style > 2000:  # High frequency emphasis
                from scipy import signal
                sos = signal.butter(2, 4000/(sr/2), btype='high', output='sos')
                styled_audio = signal.sosfilt(sos, styled_audio) * 0.7 + styled_audio * 0.3
            
            if energy_style > 0.1:  # Energy boost
                styled_audio = styled_audio * min(1.5, energy_style * 10)
            
            # Normalize
            styled_audio = styled_audio / (np.max(np.abs(styled_audio)) + 1e-8) * 0.95
            
            return styled_audio
            
        except Exception as e:
            print(f"Error converting features to audio: {e}")
            # Fallback: return original audio
            y_ref, sr = librosa.load(reference_audio_path, duration=120)
            return y_ref


# EXECUTION SCRIPT
if __name__ == "__main__":
    print("Audio Style Transfer Neural Network Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = AudioStyleTransferTrainer()
    
    # Train the neural network
    result = trainer.train_style_transfer_network(epochs=100)
    
    if result is not None:
        train_losses, test_losses = result
        print("\nTraining completed successfully!")
        print(f"Final training loss: {train_losses[-1]:.6f}")
        print(f"Final test loss: {test_losses[-1]:.6f}")
        
        # Test the model
        print("\nTesting style transfer...")
        
        # Test style transfer between DJs
        for source_dj in ['armin', 'marshmello']:
            for target_dj in ['steve', 'calvin']:
                source_path = os.path.join("audio_training", f"{source_dj}.mp3")
                if os.path.exists(source_path):
                    result = trainer.apply_neural_style_transfer(source_path, target_dj)
                    if result:
                        print(f"Created: {source_dj} -> {target_dj} style transfer")
        
        print("\nAudio Style Transfer Neural Network ready!")
        print("Model saved to: audio_style_transfer_model.pt")
    else:
        print("Training failed - please check your audio files and try again")
        print("\nTroubleshooting steps:")
        print("1. Ensure audio files exist in 'audio_training' folder")
        print("2. Check that files are valid audio formats (MP3, WAV, etc.)")
        print("3. Make sure files are not corrupted")
        print("4. Verify sufficient disk space for processing")