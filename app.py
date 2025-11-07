import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pygame
import os
from PIL import Image, ImageTk
import threading
import time
import librosa
from tensorflow.keras.models import load_model
import numpy as np

class MusicClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Genre Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Current music state
        self.current_file = None
        self.playing = False
        self.paused = False
        self.music_length = 0  # Track total music length
        
        # model load
        self.model = load_model("best_model.h5")
        
        # Create modern UI
        self.setup_ui()
        
    def setup_ui(self):
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Custom styles
        self.style.configure('Modern.TFrame', background='#2c3e50')
        self.style.configure('Title.TLabel', 
                            background='#2c3e50', 
                            foreground='#ecf0f1',
                            font=('Arial', 24, 'bold'))
        self.style.configure('Subtitle.TLabel',
                            background='#34495e',
                            foreground='#bdc3c7',
                            font=('Arial', 12))
        
        # Main container
        main_frame = ttk.Frame(self.root, style='Modern.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(header_frame, 
                               text="🎵 Music Genre Classifier", 
                               style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame,
                                  text="Upload your music file and discover its genre",
                                  style='Subtitle.TLabel')
        subtitle_label.pack(pady=(5, 0))
        
        # File selection section
        file_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        file_frame.pack(fill=tk.X, pady=20)
        
        # File selection button
        self.select_btn = tk.Button(file_frame,
                                   text="📁 Select Music File",
                                   command=self.select_file,
                                   bg='#3498db',
                                   fg='white',
                                   font=('Arial', 12, 'bold'),
                                   relief='flat',
                                   padx=20,
                                   pady=10,
                                   cursor='hand2')
        self.select_btn.pack(pady=10)
        
        # Selected file label
        self.file_label = ttk.Label(file_frame,
                                   text="No file selected",
                                   style='Subtitle.TLabel',
                                   font=('Arial', 10))
        self.file_label.pack()
        
        # Control section
        control_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        control_frame.pack(fill=tk.X, pady=20)
        
        # Progress bar with click functionality
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame,
                                           variable=self.progress_var,
                                           maximum=100,
                                           style='modern.Horizontal.TProgressbar')
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Bind click event to progress bar
        self.progress_bar.bind('<Button-1>', self.seek_music)
        
        # Time labels
        time_frame = ttk.Frame(control_frame, style='Modern.TFrame')
        time_frame.pack(fill=tk.X)
        
        self.current_time = ttk.Label(time_frame,
                                     text="0:00",
                                     style='Subtitle.TLabel')
        self.current_time.pack(side=tk.LEFT)
        
        self.total_time = ttk.Label(time_frame,
                                   text="0:00",
                                   style='Subtitle.TLabel')
        self.total_time.pack(side=tk.RIGHT)
        
        # Control buttons
        btn_frame = ttk.Frame(control_frame, style='Modern.TFrame')
        btn_frame.pack(pady=20)
        
        self.play_btn = tk.Button(btn_frame,
                                 text="▶ Play",
                                 command=self.play_music,
                                 bg='#27ae60',
                                 fg='white',
                                 font=('Arial', 11, 'bold'),
                                 relief='flat',
                                 padx=25,
                                 pady=8,
                                 state=tk.DISABLED,
                                 cursor='hand2')
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = tk.Button(btn_frame,
                                  text="⏸ Pause",
                                  command=self.pause_music,
                                  bg='#f39c12',
                                  fg='white',
                                  font=('Arial', 11, 'bold'),
                                  relief='flat',
                                  padx=25,
                                  pady=8,
                                  state=tk.DISABLED,
                                  cursor='hand2')
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame,
                                 text="⏹ Stop",
                                 command=self.stop_music,
                                 bg='#e74c3c',
                                 fg='white',
                                 font=('Arial', 11, 'bold'),
                                 relief='flat',
                                 padx=25,
                                 pady=8,
                                 state=tk.DISABLED,
                                 cursor='hand2')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Results section
        results_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        results_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Genre prediction label
        self.genre_label = ttk.Label(results_frame,
                                    text="Genre: Not predicted",
                                    style='Title.TLabel',
                                    font=('Arial', 18))
        self.genre_label.pack(pady=10)
        
        # Confidence bar
        confidence_frame = ttk.Frame(results_frame, style='Modern.TFrame')
        confidence_frame.pack(fill=tk.X, pady=10)
        
        self.confidence_label = ttk.Label(confidence_frame,
                                         text="Confidence: 0%",
                                         style='Subtitle.TLabel')
        self.confidence_label.pack()
        
        self.confidence_bar = ttk.Progressbar(confidence_frame,
                                             maximum=100,
                                             style='modern.Horizontal.TProgressbar')
        self.confidence_bar.pack(fill=tk.X, pady=5)
        
        # Start progress updater
        self.update_progress()
        
    def select_file(self):
        file_types = [
            ("Audio Files", "*.mp3 *.wav *.ogg *.flac"),
            ("MP3 Files", "*.mp3"),
            ("WAV Files", "*.wav"),
            ("All Files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select a music file",
            filetypes=file_types
        )
        
        if filename:
            self.current_file = filename
            self.file_label.config(text=os.path.basename(filename))
            self.play_btn.config(state=tk.NORMAL)
            self.genre_label.config(text="Genre: Click play to analyze")
            self.confidence_bar['value'] = 0
            self.confidence_label.config(text="Confidence: 0%")
            
            # Reset progress
            self.progress_var.set(0)
            self.current_time.config(text="0:00")
            
            # Get music length (you might want to use mutagen for accurate duration)
            self.music_length = self.get_music_length(filename)
            self.total_time.config(text=self.format_time(self.music_length))
    
    def get_music_length(self, filename):
        """Get the length of the music file in seconds"""
        try:
            # For MP3 files, you can use mutagen for accurate duration
            # For now, using a simple estimation
            sound = pygame.mixer.Sound(filename)
            return sound.get_length()
        except:
            # Fallback to a default length if we can't determine it
            return 180  # 3 minutes default
    
    def play_music(self):
        if not self.current_file:
            return
            
        try:
            if self.paused:
                # If music was paused, unpause it
                pygame.mixer.music.unpause()
                self.paused = False
                self.playing = True
            else:
                # If not playing at all, start fresh
                pygame.mixer.music.load(self.current_file)
                pygame.mixer.music.play()
                self.playing = True
                self.paused = False
                
                # Start genre prediction in a separate thread
                threading.Thread(target=self.predict_genre, daemon=True).start()
            
            self.update_button_states()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not play file: {str(e)}")
    
    def pause_music(self):
        if self.playing and not self.paused:
            pygame.mixer.music.pause()
            self.paused = True
            self.playing = False
            self.update_button_states()
    
    def stop_music(self):
        pygame.mixer.music.stop()
        self.playing = False
        self.paused = False
        self.progress_var.set(0)
        self.current_time.config(text="0:00")
        self.update_button_states()
    
    def update_button_states(self):
        """Update the state of control buttons based on current music state"""
        if self.playing and not self.paused:
            # Music is playing
            self.play_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
        elif self.paused:
            # Music is paused
            self.play_btn.config(state=tk.NORMAL, text="▶ Resume")
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
        else:
            # Music is stopped
            self.play_btn.config(state=tk.NORMAL, text="▶ Play")
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
    
    def seek_music(self, event):
        """Allow seeking by clicking on the progress bar"""
        if not self.playing or not self.current_file:
            return
        
        # Calculate the position based on click location
        progress_bar_width = self.progress_bar.winfo_width()
        click_x = event.x
        seek_percentage = click_x / progress_bar_width
        
        # Calculate new position in seconds
        new_position = seek_percentage * self.music_length
        
        # Stop and restart music at new position
        pygame.mixer.music.stop()
        pygame.mixer.music.load(self.current_file)
        pygame.mixer.music.play(start=new_position)
        self.playing = True
        self.paused = False
        
    def predict_genre(self):
        predicted_class = ""
        genres = ["blues", "classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
        
        path = self.current_file
        if not path:
            predicted_class = "Please select an audio file first."
            return
    
        y, sr = librosa.load(path, sr=22050)
        
        # Create spectrogram (same as training)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512, n_fft=2048)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        # Ensure 128x128 shape
        if log_mel.shape[1] < 128:
            log_mel = np.pad(log_mel, ((0, 0), (0, 128 - log_mel.shape[1])), constant_values=-80)
        else:
            log_mel = log_mel[:, :128]
        
        # Normalize
        epsilon = 1e-8
        log_mel_normalized = (log_mel - log_mel.mean()) / (log_mel.std() + epsilon)
        
    #  Reshape for model (add batch and channel dimensions)
        X_input = log_mel_normalized[np.newaxis, ..., np.newaxis]
        
    # Predict
        prediction = self.model.predict(X_input)
        predicted_genre = genres[np.argmax(prediction[0])]
        confidence = np.max(prediction[0]) * 100
        
    # Update UI in main thread
        self.root.after(0, lambda: self.update_prediction(predicted_genre, confidence))
    
    def update_prediction(self, genre, confidence):
        self.genre_label.config(text=f"Genre: {genre}")
        self.confidence_bar['value'] = confidence
        self.confidence_label.config(text=f"Confidence: {confidence: .2f}%")
    
    def update_progress(self):
        if self.playing and not self.paused:
            try:
                current_pos = pygame.mixer.music.get_pos() / 1000
                
                if current_pos > 0 and self.music_length > 0:
                    progress = (current_pos / self.music_length) * 100
                    self.progress_var.set(min(progress, 100))
                    
                    # Update time labels
                    current_str = self.format_time(current_pos)
                    total_str = self.format_time(self.music_length)
                    self.current_time.config(text=current_str)
                    self.total_time.config(text=total_str)
                
                # Check if music has finished playing
                if not pygame.mixer.music.get_busy() and self.playing:
                    self.playing = False
                    self.paused = False
                    self.progress_var.set(100)
                    self.update_button_states()
                    
            except Exception as e:
                print(f"Progress update error: {e}")
        
        self.root.after(100, self.update_progress)
    
    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02d}"

# Custom progress bar style
def configure_styles():
    style = ttk.Style()
    style.configure('modern.Horizontal.TProgressbar',
                   background='#3498db',
                   troughcolor='#34495e',
                   borderwidth=0,
                   lightcolor='#3498db',
                   darkcolor='#3498db')

if __name__ == "__main__":
    root = tk.Tk()
    configure_styles()
    app = MusicClassifier(root)
    root.mainloop()