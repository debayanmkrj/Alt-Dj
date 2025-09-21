import os
import json
import numpy as np
import librosa
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

class ALTDJAudioAnalyzer:
    def __init__(self, training_data_folder=".", audio_folder="audio_training"):
        """
        ALT-DJ Audio Analyzer for Famous DJ Tracks
        Uses your existing training data correlations
        """
        self.training_data_folder = training_data_folder
        self.audio_folder = audio_folder
        self.training_correlations = {}
        self.dj_audio_features = {}
        
        # Load your existing training data
        self.load_training_correlations()
    
    def load_training_correlations(self):
        """Load your existing JSON training data with audio-visual correlations"""
        print("📊 Loading training correlations from your analysis...")
        
        # Look for your analysis JSON files
        json_files = [f for f in os.listdir(self.training_data_folder) 
                     if f.startswith('complete_analysis_train_') and f.endswith('.json')]
        
        if not json_files:
            # Fallback to regular analysis files
            json_files = [f for f in os.listdir(self.training_data_folder) 
                         if f.startswith('analysis_train_') and f.endswith('.json')]
        
        if not json_files:
            print("❌ No training analysis JSON files found!")
            return
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract correlations if they exist
                if 'audio_visual_correlations' in data:
                    self.training_correlations[json_file] = data['audio_visual_correlations']
                    print(f"✅ Loaded {len(data['audio_visual_correlations'])} correlations from {json_file}")
                else:
                    print(f"⚠️  No correlations found in {json_file}")
                    
            except Exception as e:
                print(f"❌ Error loading {json_file}: {e}")
        
        total_correlations = sum(len(corrs) for corrs in self.training_correlations.values())
        print(f"📈 Total training correlations loaded: {total_correlations}")
    
    def extract_dj_audio_features(self, audio_file, window_size=3.0, hop_size=1.0):
        """
        Extract comprehensive audio features from DJ tracks
        Uses same methodology as your training data
        """
        print(f"🎵 Analyzing {audio_file}...")
        
        audio_path = os.path.join(self.audio_folder, audio_file)
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return None
        
        try:
            # Load audio (same as your training method)
            y, sr = librosa.load(audio_path, duration=300)  # First 5 minutes for now
            
            # Extract same features as training data
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Time-series features with consistent hop length
            hop_length = 512
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
            
            # Additional DJ-specific features
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
            rms_energy = librosa.feature.rms(y=y, hop_length=hop_length)
            
            # Convert to time domain
            times = librosa.frames_to_time(np.arange(spectral_centroids.shape[1]), 
                                         sr=sr, hop_length=hop_length)
            
            # Create windowed features for prediction
            features_dict = {
                'times': times,
                'spectral_centroids': spectral_centroids[0],
                'spectral_rolloff': spectral_rolloff[0],
                'spectral_bandwidth': spectral_bandwidth[0],
                'zero_crossing_rate': zero_crossing_rate[0],
                'rms_energy': rms_energy[0],
                'mfccs': mfccs,
                'chroma': chroma,
                'tempo': tempo,
                'beats': beats,
                'duration': len(y) / sr,
                'sample_rate': sr
            }
            
            # Create sliding windows for ML training
            windowed_features = self.create_feature_windows(features_dict, window_size, hop_size)
            
            print(f"✅ Extracted {len(windowed_features)} feature windows from {audio_file}")
            return windowed_features
            
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
            return None
    
    def create_feature_windows(self, features_dict, window_size=3.0, hop_size=1.0):
        """Create sliding windows of audio features for ML training"""
        times = features_dict['times']
        duration = features_dict['duration']
        
        windows = []
        window_samples = int(window_size * features_dict['sample_rate'] / 512)  # Convert to feature frames
        hop_samples = int(hop_size * features_dict['sample_rate'] / 512)
        
        for start_idx in range(0, len(times) - window_samples, hop_samples):
            end_idx = start_idx + window_samples
            
            if end_idx >= len(times):
                break
            
            # Extract window features
            window_features = {
                'start_time': float(times[start_idx]),
                'end_time': float(times[end_idx-1]),
                'duration': window_size,
                
                # Statistical features over the window
                'spectral_centroid_mean': float(np.mean(features_dict['spectral_centroids'][start_idx:end_idx])),
                'spectral_centroid_std': float(np.std(features_dict['spectral_centroids'][start_idx:end_idx])),
                'spectral_centroid_change': float(np.mean(np.abs(np.diff(features_dict['spectral_centroids'][start_idx:end_idx])))),
                
                'spectral_rolloff_mean': float(np.mean(features_dict['spectral_rolloff'][start_idx:end_idx])),
                'spectral_rolloff_std': float(np.std(features_dict['spectral_rolloff'][start_idx:end_idx])),
                'spectral_rolloff_change': float(np.mean(np.abs(np.diff(features_dict['spectral_rolloff'][start_idx:end_idx])))),
                
                'spectral_bandwidth_mean': float(np.mean(features_dict['spectral_bandwidth'][start_idx:end_idx])),
                'spectral_bandwidth_std': float(np.std(features_dict['spectral_bandwidth'][start_idx:end_idx])),
                
                'rms_energy_mean': float(np.mean(features_dict['rms_energy'][start_idx:end_idx])),
                'rms_energy_std': float(np.std(features_dict['rms_energy'][start_idx:end_idx])),
                'rms_energy_change': float(np.mean(np.abs(np.diff(features_dict['rms_energy'][start_idx:end_idx])))),
                
                'zero_crossing_rate_mean': float(np.mean(features_dict['zero_crossing_rate'][start_idx:end_idx])),
                
                # MFCC features (first 5 coefficients)
                'mfcc_1_mean': float(np.mean(features_dict['mfccs'][1, start_idx:end_idx])),
                'mfcc_2_mean': float(np.mean(features_dict['mfccs'][2, start_idx:end_idx])),
                'mfcc_3_mean': float(np.mean(features_dict['mfccs'][3, start_idx:end_idx])),
                'mfcc_4_mean': float(np.mean(features_dict['mfccs'][4, start_idx:end_idx])),
                'mfcc_5_mean': float(np.mean(features_dict['mfccs'][5, start_idx:end_idx])),
                
                # Chroma features
                'chroma_mean': float(np.mean(features_dict['chroma'][:, start_idx:end_idx])),
                'chroma_std': float(np.std(features_dict['chroma'][:, start_idx:end_idx])),
                
                'tempo': float(features_dict['tempo'])
            }
            
            windows.append(window_features)
        
        return windows
    
    def analyze_all_dj_tracks(self):
        """Analyze all DJ tracks in the audio_training folder"""
        print("🎧 Analyzing all DJ tracks...")
        
        if not os.path.exists(self.audio_folder):
            print(f"❌ Audio folder not found: {self.audio_folder}")
            return
        
        # Find audio files
        audio_files = [f for f in os.listdir(self.audio_folder) 
                      if f.endswith(('.mp3', '.wav', '.m4a', '.flac'))]
        
        if not audio_files:
            print(f"❌ No audio files found in {self.audio_folder}")
            return
        
        print(f"Found {len(audio_files)} DJ tracks: {audio_files}")
        
        for audio_file in audio_files:
            dj_name = audio_file.split('.')[0]  # Get DJ name from filename
            features = self.extract_dj_audio_features(audio_file)
            
            if features:
                self.dj_audio_features[dj_name] = features
                print(f"✅ Completed analysis for {dj_name}")
        
        print(f"🎉 Analyzed {len(self.dj_audio_features)} DJ tracks!")
    
    def build_training_dataset(self):
        """Build ML training dataset from your correlations and DJ audio features"""
        print("🏗️  Building training dataset...")
        
        if not self.training_correlations:
            print("❌ No training correlations loaded!")
            return None, None
        
        # Extract training patterns from your correlation data
        training_features = []
        training_labels = []
        
        for json_file, correlations in self.training_correlations.items():
            for correlation in correlations:
                # Extract features (same format as DJ tracks)
                feature_vector = [
                    correlation['audio_features']['spectral_centroid'],
                    correlation['audio_features']['spectral_rolloff'], 
                    correlation['audio_features']['audio_change_magnitude'],
                    correlation['correlation_strength']
                ]
                
                # Label is the interaction type
                interaction_label = correlation['interaction_type']
                
                training_features.append(feature_vector)
                training_labels.append(interaction_label)
        
        training_features = np.array(training_features)
        
        print(f"✅ Built training dataset: {training_features.shape[0]} samples, {training_features.shape[1]} features")
        print(f"Unique interactions: {len(set(training_labels))}")
        
        return training_features, training_labels
    
    def predict_dj_interactions(self, dj_name):
        """Predict what interactions a specific DJ would make on their track"""
        print(f"🔮 Predicting interactions for {dj_name}...")
        
        if dj_name not in self.dj_audio_features:
            print(f"❌ No audio features found for {dj_name}")
            return None
        
        dj_features = self.dj_audio_features[dj_name]
        
        # For now, use simple heuristic prediction (we'll replace with trained model)
        predicted_interactions = []
        
        for window in dj_features:
            # Simple prediction logic based on your training data patterns
            interaction_type = self.predict_interaction_from_features(window)
            
            prediction = {
                'timestamp': window['start_time'],
                'predicted_interaction': interaction_type,
                'confidence': self.calculate_prediction_confidence(window),
                'audio_features': window
            }
            
            predicted_interactions.append(prediction)
        
        print(f"✅ Generated {len(predicted_interactions)} interaction predictions for {dj_name}")
        return predicted_interactions
    
    def predict_interaction_from_features(self, features):
        """Simple heuristic prediction (will be replaced by trained ML model)"""
        # Based on your training data patterns
        centroid_change = features['spectral_centroid_change']
        rolloff_change = features['spectral_rolloff_change']
        energy_change = features['rms_energy_change']
        
        # High energy + high spectral change = crossfader/volume work
        if energy_change > 0.1 and (centroid_change > 1000 or rolloff_change > 2000):
            return 'right_hand-crossfader'
        
        # High spectral change = EQ work
        elif centroid_change > 500 or rolloff_change > 1000:
            return 'right_hand-eq_knobs'
        
        # High energy change = volume faders
        elif energy_change > 0.05:
            return 'right_hand-volume_faders'
        
        # Low energy, consistent = jog wheels or cue work
        elif energy_change < 0.02:
            if features['tempo'] > 120:
                return 'right_hand-hot_cues'
            else:
                return 'right_hand-jog_wheels'
        
        return 'right_hand-volume_faders'  # Default
    
    def calculate_prediction_confidence(self, features):
        """Calculate confidence in the prediction"""
        # Simple confidence based on feature stability
        stability = 1.0 - (features['spectral_centroid_std'] / max(features['spectral_centroid_mean'], 1))
        return max(0.3, min(0.95, stability))
    
    def save_analysis_results(self, output_file="altdj_analysis_results.json"):
        """Save all analysis results"""
        results = {
            'metadata': {
                'processed_at': datetime.now().isoformat(),
                'training_correlations_count': sum(len(corrs) for corrs in self.training_correlations.values()),
                'analyzed_djs': list(self.dj_audio_features.keys())
            },
            'dj_predictions': {}
        }
        
        # Generate predictions for each DJ
        for dj_name in self.dj_audio_features.keys():
            predictions = self.predict_dj_interactions(dj_name)
            if predictions:
                results['dj_predictions'][dj_name] = predictions
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Saved ALT-DJ analysis results to: {output_file}")
        return results


# EXECUTION SCRIPT
if __name__ == "__main__":
    print("🎛️ ALT-DJ Audio Analyzer - Phase 1")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ALTDJAudioAnalyzer()
    
    # Analyze all DJ tracks
    analyzer.analyze_all_dj_tracks()
    
    # Build training dataset from your existing correlations
    X_train, y_train = analyzer.build_training_dataset()
    
    if X_train is not None:
        print(f"\n📊 Training Dataset Summary:")
        print(f"Features shape: {X_train.shape}")
        print(f"Unique labels: {len(set(y_train))}")
        
        # Display interaction type distribution
        from collections import Counter
        label_counts = Counter(y_train)
        print("\nInteraction Type Distribution:")
        for interaction, count in label_counts.most_common():
            print(f"  {interaction}: {count}")
    
    # Save analysis results
    results = analyzer.save_analysis_results()
    
    print(f"\n🎯 ALT-DJ Phase 1 Complete!")
    print(f"Ready for Phase 2: Neural Network Training")
    print(f"Predicted interactions for: {list(results['dj_predictions'].keys())}")