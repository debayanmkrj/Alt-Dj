import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pygame
import cv2
import os
import json
import numpy as np
import torch
import librosa
import threading
import time
from PIL import Image, ImageTk
import glob
from collections import defaultdict
import soundfile as sf

class ALTDJDesktopApp:
    def __init__(self):
        """ALT-DJ Desktop Application with Neural Network Audio Style Transfer"""
        self.root = tk.Tk()
        self.root.title("ALT-DJ")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2E2E2E')
        
        # Initialize pygame for audio
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Application state
        self.current_track = None
        self.current_dj_style = "marshmello"
        self.is_playing = False
        self.audio_position = 0.0
        self.timeline = []
        self.current_interaction_index = 0
        
        # Video display
        self.current_video_cap = None
        self.video_thread = None
        self.video_running = False
        
        # Models and data
        self.interaction_model = None
        self.style_transfer_model = None
        self.clips_database = defaultdict(list)
        
        # DJ names
        self.dj_names = ['armin', 'calvin', 'marshmello', 'parra', 'steve']
        
        # Load models and data
        self.load_models()
        self.load_clips_database()
        
        # Create GUI
        self.create_gui()
        
        # Start audio monitoring thread
        self.audio_monitor_thread = threading.Thread(target=self.monitor_audio_playback, daemon=True)
        self.audio_monitor_thread.start()
    
    def load_models(self):
        """Load both neural networks"""
        print("Loading neural networks...")
        
        # Load interaction prediction model
        try:
            from altdj_neural_network import ALTDJNeuralNetwork
            
            if os.path.exists("altdj_trained_model.pt"):
                checkpoint = torch.load("altdj_trained_model.pt", map_location='cpu', weights_only=False)
                
                input_size = checkpoint['model_architecture']['input_size']
                num_classes = checkpoint['model_architecture']['num_classes']
                
                self.interaction_model = ALTDJNeuralNetwork(input_size, num_classes=num_classes)
                self.interaction_model.load_state_dict(checkpoint['model_state_dict'])
                self.interaction_model.eval()
                
                self.interaction_scaler = checkpoint['scaler']
                self.interaction_label_encoder = checkpoint['label_encoder']
                self.interaction_types = checkpoint['interaction_types']
                
                print("Interaction prediction model loaded")
            else:
                print("Interaction model not found")
                
        except Exception as e:
            print(f"Error loading interaction model: {e}")
        
        # Load style transfer model
        try:
            if os.path.exists("audio_style_transfer_model.pt"):
                # Only import when we actually need to use it
                self.style_transfer_trainer = None
                self.style_transfer_model = "model_exists"  # Flag that model exists
                print("Style transfer model found - will load when needed")
            else:
                print("Style transfer model not found")
                self.style_transfer_model = None
                
        except Exception as e:
            print(f"Error checking style transfer model: {e}")
            self.style_transfer_model = None
    
    def load_clips_database(self):
        """Load video clips database"""
        print("Loading video clips database...")
        
        clip_dirs = glob.glob("video_clips_train_*")
        total_clips = 0
        
        for clip_dir in clip_dirs:
            video_files = glob.glob(os.path.join(clip_dir, "*.mp4"))
            
            for video_file in video_files:
                filename = os.path.basename(video_file)
                
                # Parse interaction type from filename
                parts = filename.replace('.mp4', '').split('_')
                interaction_parts = []
                
                for i, part in enumerate(parts):
                    if part.isdigit() or (part.replace('.', '').replace('-', '').isdigit()):
                        break
                    interaction_parts.append(part)
                
                interaction_type = '_'.join(interaction_parts)
                
                if os.path.getsize(video_file) > 1000:  # Only include substantial files
                    clip_info = {
                        'filename': filename,
                        'full_path': video_file,
                        'interaction_type': interaction_type,
                        'file_size': os.path.getsize(video_file)
                    }
                    
                    self.clips_database[interaction_type].append(clip_info)
                    total_clips += 1
        
        print(f"Loaded {total_clips} video clips across {len(self.clips_database)} interaction types")
    
    def create_gui(self):
        """Create the desktop GUI"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="ALT-DJ", 
                               font=('Arial', 20, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, text= "Style Transfer + Video Visualization", 
                                  font=('Arial', 12))
        subtitle_label.pack()
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # DJ Style Selection
        ttk.Label(control_frame, text="DJ Style:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.dj_style_var = tk.StringVar(value=self.current_dj_style)
        dj_style_combo = ttk.Combobox(control_frame, textvariable=self.dj_style_var, 
                                     values=[dj.title() for dj in self.dj_names], 
                                     state="readonly", font=('Arial', 12))
        dj_style_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        dj_style_combo.bind('<<ComboboxSelected>>', self.on_dj_style_changed)
        
        # Track Selection
        ttk.Label(control_frame, text="Track:", font=('Arial', 12, 'bold')).grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        
        self.track_var = tk.StringVar()
        self.track_combo = ttk.Combobox(control_frame, textvariable=self.track_var, 
                                       values=[], state="readonly", font=('Arial', 12))
        self.track_combo.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        self.load_available_tracks()
        
        # Buttons
        ttk.Button(control_frame, text="Load Custom Track", 
                  command=self.load_custom_track).grid(row=0, column=4, padx=(0, 10))
        
        ttk.Button(control_frame, text="Analyze & Style Transfer", 
                  command=self.analyze_and_process).grid(row=0, column=5, padx=(0, 10))
        
        # Audio Controls
        audio_control_frame = ttk.Frame(control_frame)
        audio_control_frame.grid(row=1, column=0, columnspan=6, sticky=tk.W, pady=(10, 0))
        
        self.play_button = ttk.Button(audio_control_frame, text="Play", 
                                     command=self.toggle_playback, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(audio_control_frame, text="Stop", 
                  command=self.stop_playback).pack(side=tk.LEFT, padx=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(audio_control_frame, variable=self.progress_var, 
                                          length=300, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 10))
        
        self.time_label = ttk.Label(audio_control_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.LEFT)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video and Timeline
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video display
        video_frame = ttk.LabelFrame(left_panel, text="DJ Hand Interactions", padding=10)
        video_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.video_label = ttk.Label(video_frame, text="Video will appear here", 
                                    background='black', foreground='white', 
                                    font=('Arial', 14), anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        # Configure minimum size for video display area
        video_frame.configure(height=300)
        self.video_label.configure(width=400)
        
        # Current interaction info
        interaction_info_frame = ttk.LabelFrame(left_panel, text="Current Interaction", padding=10)
        interaction_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.interaction_info_text = tk.Text(interaction_info_frame, height=4, wrap=tk.WORD)
        self.interaction_info_text.pack(fill=tk.X)
        self.interaction_info_text.insert(tk.END, "No interaction selected")
        
        # Timeline
        timeline_frame = ttk.LabelFrame(left_panel, text="Interaction Timeline", padding=10)
        timeline_frame.pack(fill=tk.BOTH, expand=True)
        
        # Timeline listbox with scrollbar
        timeline_scroll_frame = ttk.Frame(timeline_frame)
        timeline_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.timeline_listbox = tk.Listbox(timeline_scroll_frame, font=('Courier', 10))
        timeline_scrollbar = ttk.Scrollbar(timeline_scroll_frame, orient=tk.VERTICAL, 
                                         command=self.timeline_listbox.yview)
        self.timeline_listbox.configure(yscrollcommand=timeline_scrollbar.set)
        
        self.timeline_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        timeline_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.timeline_listbox.bind('<Double-Button-1>', self.on_timeline_select)
        
        # Right panel - Analysis and Logs
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right_panel.configure(width=400)
        
        # Analysis status
        status_frame = ttk.LabelFrame(right_panel, text="Analysis Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=8, width=40)
        self.status_text.pack(fill=tk.X)
        self.status_text.insert(tk.END, "Ready for analysis...\n")
        
        # Model status
        model_frame = ttk.LabelFrame(right_panel, text="Neural Networks", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        interaction_status = "Loaded" if self.interaction_model else "Not Available"
        style_status = "Loaded" if self.style_transfer_model else "Not Available"
        
        ttk.Label(model_frame, text=f"Interaction Prediction: {interaction_status}").pack(anchor=tk.W)
        ttk.Label(model_frame, text=f"Style Transfer: {style_status}").pack(anchor=tk.W)
        ttk.Label(model_frame, text=f"Video Clips: {sum(len(clips) for clips in self.clips_database.values())}").pack(anchor=tk.W)
        
        # Train Style Transfer button
        if not self.style_transfer_model:
            ttk.Button(model_frame, text="Train Style Transfer NN", 
                      command=self.train_style_transfer).pack(pady=(10, 0))
    
    def load_available_tracks(self):
        """Load available DJ tracks"""
        tracks = []
        audio_folder = "audio_training"
        
        if os.path.exists(audio_folder):
            for file in os.listdir(audio_folder):
                if file.endswith(('.mp3', '.wav', '.m4a')):
                    track_name = file.split('.')[0]
                    tracks.append(track_name.title())
        
        self.track_combo['values'] = tracks
        if tracks:
            self.track_combo.set(tracks[0])
    
    def load_custom_track(self):
        """Load a custom audio track"""
        file_path = filedialog.askopenfilename(
            title="Select Audio Track",
            filetypes=[("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg")]
        )
        
        if file_path:
            self.custom_track_path = file_path
            track_name = os.path.basename(file_path)
            self.track_var.set(f"Custom: {track_name}")
            self.log(f"Loaded custom track: {track_name}")
    
    def on_dj_style_changed(self, event=None):
        """Handle DJ style selection change"""
        self.current_dj_style = self.dj_style_var.get().lower()
        self.log(f"Selected DJ style: {self.current_dj_style}")
    
    def analyze_and_process(self):
        """Analyze track and apply style transfer"""
        if not self.track_var.get():
            messagebox.showerror("Error", "Please select a track first")
            return
        
        if not self.interaction_model:
            messagebox.showerror("Error", "Interaction prediction model not loaded")
            return
        
        # Run in separate thread to avoid blocking GUI
        analysis_thread = threading.Thread(target=self._analyze_and_process_thread, daemon=True)
        analysis_thread.start()
    
    def _analyze_and_process_thread(self):
        """Background thread for analysis and processing"""
        try:
            self.log("Starting analysis and style transfer...")
            
            # Determine audio file path
            if self.track_var.get().startswith("Custom:"):
                audio_path = getattr(self, 'custom_track_path', None)
                if not audio_path:
                    self.log("ERROR: Custom track path not found")
                    return
            else:
                track_name = self.track_var.get().lower()
                audio_path = os.path.join("audio_training", f"{track_name}.mp3")
                
                if not os.path.exists(audio_path):
                    self.log(f"ERROR: Audio file not found: {audio_path}")
                    return
            
            # Step 1: Predict interactions
            self.log("Predicting DJ interactions with neural network...")
            predictions = self.predict_interactions(audio_path)
            
            if not predictions:
                self.log("ERROR: Failed to generate interaction predictions")
                return
            
            self.log(f"Generated {len(predictions)} interaction predictions")
            
            # Step 2: Apply style transfer
            styled_audio_path = audio_path
            style_transfer_applied = False
            
            if self.style_transfer_model:
                self.log(f"Applying {self.current_dj_style} neural style transfer...")
                styled_audio_path = self.apply_neural_style_transfer(audio_path, predictions)
                
                if styled_audio_path and styled_audio_path != audio_path:
                    self.log("Neural style transfer completed!")
                    style_transfer_applied = True
                else:
                    self.log("Style transfer failed, using original audio")
                    styled_audio_path = audio_path
            else:
                self.log("Style transfer model not available, using original audio")
            
            # Step 3: Build timeline
            self.timeline = self.build_timeline_with_videos(predictions)
            
            # Step 4: Update GUI
            self.root.after(0, self.update_gui_after_analysis, styled_audio_path, style_transfer_applied)
            
        except Exception as e:
            self.log(f"ERROR: Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    def predict_interactions(self, audio_path):
        """Predict DJ interactions using the trained neural network"""
        try:
            # Load and analyze audio
            y, sr = librosa.load(audio_path, duration=180)  # 3 minutes
            
            if len(y) == 0:
                return None
            
            hop_length = 512
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
            rms_energy = librosa.feature.rms(y=y, hop_length=hop_length)
            
            times = librosa.frames_to_time(np.arange(spectral_centroids.shape[1]), sr=sr, hop_length=hop_length)
            
            predictions = []
            window_size = 3.0
            hop_size = 2.0
            
            for i in range(0, len(times) - int(window_size * sr / hop_length), int(hop_size * sr / hop_length)):
                end_idx = i + int(window_size * sr / hop_length)
                
                if end_idx >= len(times):
                    break
                
                # Extract window features
                centroid_mean = float(np.mean(spectral_centroids[0, i:end_idx]))
                rolloff_mean = float(np.mean(spectral_rolloff[0, i:end_idx]))
                centroid_change = float(np.mean(np.abs(np.diff(spectral_centroids[0, i:end_idx]))))
                rolloff_change = float(np.mean(np.abs(np.diff(spectral_rolloff[0, i:end_idx]))))
                
                audio_change_magnitude = centroid_change + rolloff_change
                
                features = np.array([[
                    centroid_mean,
                    rolloff_mean,
                    audio_change_magnitude,
                    0.7,  # Default confidence
                    0.7   # Default interaction confidence
                ]])
                
                # Scale and predict
                features_scaled = self.interaction_scaler.transform(features)
                
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features_scaled)
                    outputs = self.interaction_model(features_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = torch.max(probabilities).item()
                    
                    interaction_type = self.interaction_types[predicted_class]
                    
                    prediction = {
                        'timestamp': float(times[i]),
                        'interaction_type': interaction_type,
                        'confidence': float(confidence)
                    }
                    
                    predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            self.log(f"Error in interaction prediction: {e}")
            return None
    
    def apply_neural_style_transfer(self, audio_path, predictions):
        """Apply neural network style transfer"""
        if self.style_transfer_model != "model_exists":
            self.log("Style transfer model not available")
            return audio_path
        
        try:
            # Load the trainer only when we actually need it
            if not hasattr(self, 'style_transfer_trainer') or self.style_transfer_trainer is None:
                from audio_style_transfer_nn import AudioStyleTransferTrainer
                self.style_transfer_trainer = AudioStyleTransferTrainer()
                if not self.style_transfer_trainer.load_model():
                    self.log("Failed to load style transfer model")
                    return audio_path
                self.log("Style transfer model loaded successfully")
            
            self.log(f"Applying {self.current_dj_style} neural style transfer...")
            
            output_folder = "processed_audio"
            os.makedirs(output_folder, exist_ok=True)
            
            output_path = os.path.join(output_folder, 
                                     f"neural_{self.current_dj_style}_{int(time.time())}.wav")
            
            # Use the style transfer trainer's method
            styled_path = self.style_transfer_trainer.apply_neural_style_transfer(
                audio_path, self.current_dj_style, output_path
            )
            
            if styled_path and os.path.exists(styled_path):
                file_size = os.path.getsize(styled_path)
                self.log(f"Style transfer completed: {os.path.basename(styled_path)} ({file_size} bytes)")
                return styled_path
            else:
                self.log("Style transfer failed - using original audio")
                return audio_path
            
        except Exception as e:
            self.log(f"Error in neural style transfer: {e}")
            import traceback
            traceback.print_exc()
            return audio_path
    
    def build_timeline_with_videos(self, predictions):
        """Build timeline with matching video clips"""
        timeline = []
        
        for prediction in predictions:
            interaction_type = prediction['interaction_type']
            timestamp = prediction['timestamp']
            confidence = prediction['confidence']
            
            # Find matching video clip
            video_clip = self.find_matching_video_clip(interaction_type)
            
            timeline_item = {
                'timestamp': timestamp,
                'interaction_type': interaction_type,
                'confidence': confidence,
                'video_clip': video_clip,
                'has_video': video_clip is not None
            }
            
            timeline.append(timeline_item)
        
        return timeline
    
    def find_matching_video_clip(self, interaction_type):
        """Find the best matching video clip for an interaction type"""
        # Direct match first
        if interaction_type in self.clips_database:
            clips = self.clips_database[interaction_type]
            if clips:
                # Return clip with largest file size (usually better quality)
                return max(clips, key=lambda x: x['file_size'])
        
        # Fuzzy matching
        interaction_clean = interaction_type.replace('_', '').replace('-', '').lower()
        
        best_match = None
        best_score = 0
        
        for available_type, clips in self.clips_database.items():
            if not clips:
                continue
                
            available_clean = available_type.replace('_', '').replace('-', '').lower()
            
            if interaction_clean in available_clean or available_clean in interaction_clean:
                score = min(len(interaction_clean), len(available_clean)) / max(len(interaction_clean), len(available_clean))
                if score > best_score:
                    best_score = score
                    best_match = max(clips, key=lambda x: x['file_size'])
        
        return best_match
    
    def update_gui_after_analysis(self, styled_audio_path, style_transfer_applied=False):
        """Update GUI after analysis is complete"""
        # Update timeline listbox
        self.timeline_listbox.delete(0, tk.END)
        
        for i, item in enumerate(self.timeline):
            timestamp = item['timestamp']
            interaction = item['interaction_type']
            confidence = item['confidence']
            video_status = "V" if item['has_video'] else "-"
            
            line = f"{timestamp:6.1f}s [{video_status}] {interaction} ({confidence:.2f})"
            self.timeline_listbox.insert(tk.END, line)
        
        # Load audio
        if styled_audio_path and os.path.exists(styled_audio_path):
            try:
                pygame.mixer.music.load(styled_audio_path)
                self.current_track = styled_audio_path
                self.play_button['state'] = tk.NORMAL
                
                if style_transfer_applied:
                    self.log(f"Audio loaded with {self.current_dj_style} style transfer: {os.path.basename(styled_audio_path)}")
                else:
                    self.log(f"Audio loaded (original): {os.path.basename(styled_audio_path)}")
                
                # Get audio duration
                y, sr = librosa.load(styled_audio_path, duration=None)
                self.audio_duration = len(y) / sr
                
            except Exception as e:
                self.log(f"Error loading audio: {e}")
        
        status_msg = "Analysis complete!"
        if style_transfer_applied:
            status_msg += f" Style transfer applied ({self.current_dj_style})"
        else:
            status_msg += " No style transfer applied"
            
        self.log(status_msg)
    
    def toggle_playback(self):
        """Toggle audio playback"""
        if self.is_playing:
            pygame.mixer.music.pause()
            self.play_button['text'] = "Resume"
            self.is_playing = False
        else:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.unpause()
            else:
                pygame.mixer.music.play()
            self.play_button['text'] = "Pause"
            self.is_playing = True
    
    def stop_playback(self):
        """Stop audio playback"""
        pygame.mixer.music.stop()
        self.play_button['text'] = "Play"
        self.is_playing = False
        self.audio_position = 0.0
        self.progress_var.set(0)
    
    def on_timeline_select(self, event):
        """Handle timeline item selection"""
        selection = self.timeline_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index < len(self.timeline):
            item = self.timeline[index]
            self.current_interaction_index = index
            
            # Update interaction info
            self.interaction_info_text.delete(1.0, tk.END)
            info_text = f"Timestamp: {item['timestamp']:.1f}s\n"
            info_text += f"Interaction: {item['interaction_type']}\n"
            info_text += f"Confidence: {item['confidence']:.3f}\n"
            info_text += f"Video Available: {'Yes' if item['has_video'] else 'No'}"
            self.interaction_info_text.insert(1.0, info_text)
            
            # Show video if available
            if item['has_video'] and item['video_clip']:
                self.show_video_clip(item['video_clip']['full_path'])
            else:
                self.video_label['text'] = f"No video for\n{item['interaction_type']}"
            
            # Seek audio to this timestamp
            if self.current_track:
                try:
                    # Stop current playback
                    pygame.mixer.music.stop()
                    
                    # Reload and seek (pygame doesn't support seeking, so we approximate)
                    pygame.mixer.music.load(self.current_track)
                    self.audio_position = item['timestamp']
                    
                    if self.is_playing:
                        pygame.mixer.music.play()
                    
                except Exception as e:
                    self.log(f"Error seeking audio: {e}")
    
    def show_video_clip(self, video_path):
        """Display video clip"""
        try:
            # Stop current video
            if self.current_video_cap:
                self.current_video_cap.release()
                self.video_running = False
            
            # Load new video
            self.current_video_cap = cv2.VideoCapture(video_path)
            
            if not self.current_video_cap.isOpened():
                self.log(f"Could not open video: {video_path}")
                self.video_label['text'] = f"Video error:\n{os.path.basename(video_path)}"
                return
            
            self.video_running = True
            
            # Start video playback thread
            if self.video_thread and self.video_thread.is_alive():
                self.video_running = False
                self.video_thread.join(timeout=1.0)
            
            self.video_thread = threading.Thread(target=self.video_playback_loop, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            self.log(f"Error showing video: {e}")
            self.video_label['text'] = f"Video error:\n{str(e)[:50]}"
    
    def video_playback_loop(self):
        """Video playback loop"""
        try:
            while self.video_running and self.current_video_cap:
                ret, frame = self.current_video_cap.read()
                
                if not ret:
                    # Loop video
                    self.current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Resize frame to fit display
                height, width = frame.shape[:2]
                max_width, max_height = 400, 300
                
                if width > max_width or height > max_height:
                    scale = min(max_width/width, max_height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update label
                self.root.after(0, self.update_video_frame, photo)
                
                # Control frame rate
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            self.log(f"Video playback error: {e}")
    
    def update_video_frame(self, photo):
        """Update video frame in GUI"""
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo  # Keep reference
    
    def monitor_audio_playback(self):
        """Monitor audio playback position"""
        while True:
            try:
                if self.is_playing and hasattr(self, 'audio_duration'):
                    # Update progress bar and time
                    progress = (self.audio_position / self.audio_duration) * 100
                    self.progress_var.set(progress)
                    
                    current_time = int(self.audio_position)
                    total_time = int(self.audio_duration)
                    time_text = f"{current_time//60:02d}:{current_time%60:02d} / {total_time//60:02d}:{total_time%60:02d}"
                    
                    self.root.after(0, lambda: self.time_label.config(text=time_text))
                    
                    # Auto-advance timeline
                    self.auto_advance_timeline()
                    
                    self.audio_position += 1.0
                
                time.sleep(1.0)
                
            except Exception as e:
                # Silently handle monitoring errors
                time.sleep(1.0)
    
    def auto_advance_timeline(self):
        """Automatically advance timeline based on audio position"""
        if not self.timeline:
            return
        
        current_time = self.audio_position
        
        # Find closest timeline item
        closest_index = 0
        closest_diff = float('inf')
        
        for i, item in enumerate(self.timeline):
            diff = abs(item['timestamp'] - current_time)
            if diff < closest_diff:
                closest_diff = diff
                closest_index = i
        
        # Update if we're close enough and it's a different item
        if closest_diff < 2.0 and closest_index != self.current_interaction_index:
            self.current_interaction_index = closest_index
            
            # Update GUI
            self.root.after(0, self.highlight_timeline_item, closest_index)
    
    def highlight_timeline_item(self, index):
        """Highlight timeline item"""
        self.timeline_listbox.selection_clear(0, tk.END)
        self.timeline_listbox.selection_set(index)
        self.timeline_listbox.see(index)
        
        # Update interaction info and video
        if index < len(self.timeline):
            item = self.timeline[index]
            
            self.interaction_info_text.delete(1.0, tk.END)
            info_text = f"Timestamp: {item['timestamp']:.1f}s\n"
            info_text += f"Interaction: {item['interaction_type']}\n"
            info_text += f"Confidence: {item['confidence']:.3f}\n"
            info_text += f"Video Available: {'Yes' if item['has_video'] else 'No'}"
            self.interaction_info_text.insert(1.0, info_text)
            
            if item['has_video'] and item['video_clip']:
                self.show_video_clip(item['video_clip']['full_path'])
    
    def train_style_transfer(self):
        """Train the style transfer neural network"""
        def train_thread():
            try:
                self.log("Training Audio Style Transfer Neural Network...")
                
                from audio_style_transfer_nn import AudioStyleTransferTrainer
                trainer = AudioStyleTransferTrainer()
                
                train_losses, test_losses = trainer.train_style_transfer_network(epochs=50)
                
                if train_losses:
                    self.log("Style transfer training completed!")
                    
                    # Reload the model
                    if trainer.load_model():
                        self.style_transfer_model = trainer.model
                        self.style_transfer_trainer = trainer
                        self.root.after(0, self.update_model_status)
                    
                else:
                    self.log("Training failed!")
                    
            except Exception as e:
                self.log(f"Training error: {e}")
        
        training_thread = threading.Thread(target=train_thread, daemon=True)
        training_thread.start()
    
    def update_model_status(self):
        """Update model status in GUI"""
        messagebox.showinfo("Training Complete", "Audio Style Transfer Neural Network trained successfully!")
    
    def log(self, message):
        """Log message to status text"""
        timestamp = time.strftime("%H:%M:%S")
        self.root.after(0, self._log_message, f"[{timestamp}] {message}")
    
    def _log_message(self, message):
        """Thread-safe logging"""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
    
    def run(self):
        """Start the application"""
        print("Starting ALT-DJ Desktop Application...")
        self.root.mainloop()


# EXECUTION SCRIPT
if __name__ == "__main__":
    try:
        app = ALTDJDesktopApp()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()