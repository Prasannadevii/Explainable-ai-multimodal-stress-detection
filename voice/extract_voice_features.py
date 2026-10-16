import numpy as np
import librosa
import json
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
import csv
from datetime import datetime

class VoiceFeatureExtractor:
    """
    Extracts acoustic features from voice recordings that are indicative of stress.
    Features include: pitch, energy, speech rate, formants, etc.
    """
    
    def __init__(self):
        self.features_list = []
    
    def extract_features(self, audio_file, metadata_file=None):
        """
        Extract features from a single audio file
        """
        try:
            # Load audio
            y, sr = librosa.load(str(audio_file), sr=None)
            
            # Extract features
            features = {
                'file': audio_file.name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Load metadata if available
            if metadata_file and metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                features.update(metadata)
            
            # 1. Fundamental Frequency (Pitch) - typically lower under stress
            f0 = self._extract_f0(y, sr)
            features['f0_mean'] = np.mean(f0)
            features['f0_std'] = np.std(f0)
            features['f0_min'] = np.min(f0)
            features['f0_max'] = np.max(f0)
            
            # 2. Spectral Features
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            features['spectral_centroid_mean'] = np.mean(
                librosa.feature.spectral_centroid(y=y, sr=sr)
            )
            features['spectral_rolloff_mean'] = np.mean(
                librosa.feature.spectral_rolloff(y=y, sr=sr)
            )
            
            # 3. Energy - typically increases under stress
            energy = np.sum(y**2)
            features['total_energy'] = energy
            features['energy_mean'] = np.mean(y**2)
            features['energy_std'] = np.std(y**2)
            
            # 4. Zero Crossing Rate - speech rate indicator
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 5. MFCC (Mel-Frequency Cepstral Coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i, coeff in enumerate(mfcc):
                features[f'mfcc_{i}_mean'] = np.mean(coeff)
                features[f'mfcc_{i}_std'] = np.std(coeff)
            
            # 6. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            # 7. Tempogram - speech rate/rhythm
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            features['onset_strength_mean'] = np.mean(onset_env)
            features['onset_strength_std'] = np.std(onset_env)
            
            # 8. RMS Energy over time
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            features['rms_max'] = np.max(rms)
            
            # 9. Duration
            duration = librosa.get_duration(y=y, sr=sr)
            features['duration'] = duration
            
            return features
        
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            return None
    
    def _extract_f0(self, y, sr, fmin=80, fmax=400):
        """
        Extract fundamental frequency using librosa
        """
        try:
            f0 = librosa.yin(y, fmin=fmin, fmax=fmax, trough_threshold=0.1)
            # Remove unvoiced frames (NaN values)
            f0 = f0[~np.isnan(f0)]
            if len(f0) == 0:
                f0 = np.array([0])
            return f0
        except:
            return np.array([0])
    
    def process_dataset(self, data_dir, output_csv="voice_features_dataset.csv"):
        """
        Process all audio files in the dataset directory and save features to CSV
        """
        data_dir = Path(data_dir)
        all_features = []
        
        print("Processing voice data files...")
        
        # Find all wav files
        wav_files = list(data_dir.glob("*/*/*.wav"))
        print(f"Found {len(wav_files)} audio files to process")
        
        for idx, wav_file in enumerate(wav_files):
            print(f"Processing {idx + 1}/{len(wav_files)}: {wav_file.name}")
            
            json_file = wav_file.with_suffix('.json')
            features = self.extract_features(wav_file, json_file)
            
            if features:
                all_features.append(features)
        
        # Save to CSV
        if all_features:
            keys = all_features[0].keys()
            
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_features)
            
            print(f"\n✅ Features saved to {output_csv}")
            print(f"Total samples: {len(all_features)}")
            
            # Print statistics by level
            self._print_statistics(all_features)
        else:
            print("No features extracted!")
    
    def _print_statistics(self, features_list):
        """
        Print statistics of extracted features by stress level
        """
        print("\n" + "="*60)
        print("FEATURE STATISTICS BY STRESS LEVEL")
        print("="*60)
        
        for level in [1, 2, 3]:
            level_features = [f for f in features_list if f.get('level') == level]
            if level_features:
                print(f"\nLevel {level} (n={len(level_features)})")
                print("-" * 40)
                
                # Calculate means for key features
                f0_means = [f['f0_mean'] for f in level_features if f['f0_mean'] > 0]
                if f0_means:
                    print(f"  Pitch (F0):        {np.mean(f0_means):.2f} Hz")
                
                print(f"  Energy:           {np.mean([f['energy_mean'] for f in level_features]):.4f}")
                print(f"  RMS:              {np.mean([f['rms_mean'] for f in level_features]):.4f}")
                print(f"  ZCR:              {np.mean([f['zcr_mean'] for f in level_features]):.4f}")
                print(f"  Spectral Centroid:{np.mean([f['spectral_centroid_mean'] for f in level_features]):.2f} Hz")


if __name__ == "__main__":
    from pathlib import Path as _Path
    _PROJECT_ROOT = _Path(__file__).resolve().parent.parent
    _VOICE_DATA_DIR = _PROJECT_ROOT / "data" / "voice" / "voice_stress_data"
    _OUTPUT_CSV = _PROJECT_ROOT / "data" / "voice" / "voice_features_dataset.csv"

    extractor = VoiceFeatureExtractor()
    
    # Process all data
    extractor.process_dataset(
        data_dir=str(_VOICE_DATA_DIR),
        output_csv=str(_OUTPUT_CSV)
    )
    
    print("\n✅ Feature extraction complete!")
    print(f"You can now use '{_OUTPUT_CSV}' for training your ML model")
