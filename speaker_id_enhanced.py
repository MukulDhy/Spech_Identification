import torch
import torchaudio
import numpy as np
import librosa
import webrtcvad
import noisereduce as nr
from speechbrain.inference.speaker import SpeakerRecognition  # ← FIXED IMPORT
from pydub import AudioSegment
from pathlib import Path
import tempfile
import io
import logging
from typing import Optional, Tuple, Dict, List
import asyncio
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import os

# Set environment variable to avoid symlink issues on Windows
os.environ['SPEECHBRAIN_CACHE_DIR'] = './speechbrain_cache'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSpeakerIdentifier:
    def __init__(self, embedding_folder="family_embeddings"):
        self.embedding_folder = Path(embedding_folder)
        self.embedding_folder.mkdir(exist_ok=True)
        
        # Create cache directory
        cache_dir = Path("speechbrain_cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Load models with Windows-friendly settings
        logger.info("Loading SpeechBrain ECAPA-TDNN model...")
        try:
            self.spkrec = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir="./speechbrain_cache/spkrec",
                run_opts={"device": "cpu"}  # Force CPU to avoid CUDA issues
            )
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain model: {e}")
            raise
        
        # Load Silero VAD
        logger.info("Loading Silero VAD model...")
        try:
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', 
                model='silero_vad', 
                force_reload=False
            )
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            raise
        
        # WebRTC VAD for additional validation
        try:
            self.webrtc_vad = webrtcvad.Vad()
            self.webrtc_vad.set_mode(3)  # Most aggressive mode
        except Exception as e:
            logger.warning(f"WebRTC VAD not available: {e}")
            self.webrtc_vad = None
        
        # Load family database
        self.family_db = self.load_family_embeddings()
        
        # Audio processing settings
        self.target_sample_rate = 16000
        self.min_speech_duration = 0.5  # Minimum 0.5 seconds
        self.max_audio_length = 30  # Maximum 30 seconds

    # ... rest of the methods remain the same ...
    
    def preprocess_audio(self, audio_data: bytes) -> Tuple[np.ndarray, int]:
        """Enhanced audio preprocessing with noise reduction"""
        try:
            # Convert bytes to audio
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            
            # Convert to mono and target sample rate
            audio_segment = audio_segment.set_channels(1).set_frame_rate(self.target_sample_rate)
            
            # Convert to numpy array
            audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Normalize
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Apply noise reduction
            try:
                audio_array = nr.reduce_noise(y=audio_array, sr=self.target_sample_rate)
            except Exception as e:
                logger.warning(f"Noise reduction failed: {e}")
            
            # Limit audio length
            max_samples = self.max_audio_length * self.target_sample_rate
            if len(audio_array) > max_samples:
                audio_array = audio_array[:max_samples]
                
            return audio_array, self.target_sample_rate
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            raise ValueError(f"Failed to process audio: {e}")

    def detect_speech_segments(self, audio_array: np.ndarray, sample_rate: int) -> List[Dict]:
        """Advanced speech detection using multiple VAD methods"""
        try:
            # Method 1: Silero VAD
            audio_tensor = torch.from_numpy(audio_array)
            speech_timestamps = self.vad_utils[0](audio_tensor, self.vad_model)
            
            # Convert to segments
            speech_segments = []
            for segment in speech_timestamps:
                start_time = segment['start'] / sample_rate
                end_time = segment['end'] / sample_rate
                duration = end_time - start_time
                
                if duration >= self.min_speech_duration:
                    speech_segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'duration': duration,
                        'confidence': 0.9,
                        'method': 'silero'
                    })
            
            return speech_segments
            
        except Exception as e:
            logger.error(f"Speech detection error: {e}")
            return []

    def extract_enhanced_embedding(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract speaker embedding with enhanced features"""
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            
            # Extract ECAPA-TDNN embedding
            with torch.no_grad():
                embedding = self.spkrec.encode_batch(audio_tensor)
                base_embedding = embedding.squeeze().cpu().numpy()
            
            # Additional features using librosa
            try:
                # MFCC features
                mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
                mfcc_mean = np.mean(mfccs, axis=1)
                
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=sample_rate)
                
                # Combine features
                additional_features = np.concatenate([
                    mfcc_mean,
                    [np.mean(spectral_centroids)],
                    [np.mean(spectral_rolloff)]
                ], axis=0)
                
                # Normalize additional features
                if np.linalg.norm(additional_features) > 0:
                    additional_features = additional_features / np.linalg.norm(additional_features)
                
                # Combine: 90% ECAPA-TDNN, 10% additional features
                enhanced_embedding = np.concatenate([
                    base_embedding * 0.9,
                    additional_features * 0.1
                ])
                
            except Exception as e:
                logger.warning(f"Additional features extraction failed: {e}")
                enhanced_embedding = base_embedding
            
            # Normalize final embedding
            if np.linalg.norm(enhanced_embedding) > 0:
                enhanced_embedding = enhanced_embedding / np.linalg.norm(enhanced_embedding)
            
            return enhanced_embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            raise ValueError(f"Failed to extract embedding: {e}")

    def load_family_embeddings(self) -> Dict[str, np.ndarray]:
        """Load pre-computed family member embeddings"""
        db = {}
        for file in self.embedding_folder.glob("*.npy"):
            name = file.stem
            embedding = np.load(file)
            db[name] = embedding
            logger.info(f"Loaded embedding for {name}")
        return db

    def match_speaker_advanced(self, embedding: np.ndarray, threshold: float = 0.75) -> Tuple[str, float]:
        """Advanced speaker matching with multiple similarity metrics"""
        if not self.family_db:
            return "unknown", 0.0
        
        scores = {}
        
        for name, db_embedding in self.family_db.items():
            # Ensure embeddings have same dimension
            min_dim = min(len(embedding), len(db_embedding))
            emb1 = embedding[:min_dim]
            emb2 = db_embedding[:min_dim]
            
            # Cosine similarity
            try:
                cosine_sim = 1 - cosine(emb1, emb2)
            except:
                cosine_sim = 0.0
            
            # Pearson correlation
            try:
                pearson_corr, _ = pearsonr(emb1, emb2)
                if np.isnan(pearson_corr):
                    pearson_corr = 0.0
            except:
                pearson_corr = 0.0
            
            # Dot product
            try:
                dot_product = np.dot(emb1, emb2)
            except:
                dot_product = 0.0
            
            # Weighted combination
            combined_score = (cosine_sim * 0.6 + pearson_corr * 0.3 + dot_product * 0.1)
            scores[name] = combined_score
        
        best_match = max(scores, key=scores.get)
        best_score = scores[best_match]
        
        if best_score < threshold:
            return "unknown", float(best_score)
        
        return best_match, float(best_score)

    async def identify_speaker(self, audio_data: bytes) -> Dict:
        """Main identification method"""
        try:
            # Preprocess audio
            audio_array, sample_rate = self.preprocess_audio(audio_data)
            
            # Detect speech segments
            speech_segments = self.detect_speech_segments(audio_array, sample_rate)
            
            if not speech_segments:
                return {
                    "speaker": "no_speech_detected",
                    "confidence": 0.0,
                    "segments": [],
                    "status": "no_speech"
                }
            
            # Process each segment
            results = []
            for segment in speech_segments:
                start_sample = segment['start']
                end_sample = segment['end']
                segment_audio = audio_array[start_sample:end_sample]
                
                if len(segment_audio) < self.min_speech_duration * sample_rate:
                    continue
                
                # Extract embedding
                embedding = self.extract_enhanced_embedding(segment_audio, sample_rate)
                
                # Match speaker
                speaker, confidence = self.match_speaker_advanced(embedding)
                
                results.append({
                    "speaker": speaker,
                    "confidence": confidence,
                    "start_time": segment['start'] / sample_rate,
                    "end_time": segment['end'] / sample_rate,
                    "duration": segment['duration']
                })
            
            if not results:
                return {
                    "speaker": "speech_too_short",
                    "confidence": 0.0,
                    "segments": [],
                    "status": "insufficient_audio"
                }
            
            # Return best result
            best_result = max(results, key=lambda x: x['confidence'])
            
            return {
                "speaker": best_result["speaker"],
                "confidence": best_result["confidence"],
                "segments": results,
                "status": "success",
                "total_segments": len(results)
            }
            
        except Exception as e:
            logger.error(f"Speaker identification error: {e}")
            return {
                "speaker": "error",
                "confidence": 0.0,
                "segments": [],
                "status": f"error: {str(e)}"
            }

    def create_speaker_profile(self, name: str, audio_files: List[str]) -> bool:
        """Create speaker profile from multiple audio files"""
        try:
            embeddings = []
            processed_files = 0
            
            for audio_file in audio_files:
                try:
                    with open(audio_file, 'rb') as f:
                        audio_data = f.read()
                    
                    audio_array, sample_rate = self.preprocess_audio(audio_data)
                    
                    # Only use segments with clear speech
                    speech_segments = self.detect_speech_segments(audio_array, sample_rate)
                    
                    for segment in speech_segments:
                        start_sample = segment['start']
                        end_sample = segment['end']
                        segment_audio = audio_array[start_sample:end_sample]
                        
                        if len(segment_audio) >= self.min_speech_duration * sample_rate:
                            embedding = self.extract_enhanced_embedding(segment_audio, sample_rate)
                            embeddings.append(embedding)
                    
                    processed_files += 1
                    if processed_files % 10 == 0:
                        logger.info(f"Processed {processed_files}/{len(audio_files)} files for {name}")
                        
                except Exception as e:
                    logger.warning(f"Skipping file {audio_file}: {e}")
                    continue
            
            if not embeddings:
                logger.error(f"No valid embeddings found for {name}")
                return False
            
            # Create averaged embedding with outlier removal
            embeddings = np.array(embeddings)
            
            # Remove outliers using median absolute deviation
            median_emb = np.median(embeddings, axis=0)
            mad = np.median(np.abs(embeddings - median_emb), axis=0)
            
            # Keep embeddings within 2 MAD of median
            valid_mask = np.all(np.abs(embeddings - median_emb) <= 2 * (mad + 1e-10), axis=1)
            filtered_embeddings = embeddings[valid_mask]
            
            if len(filtered_embeddings) == 0:
                filtered_embeddings = embeddings  # Fallback
            
            # Create final embedding
            final_embedding = np.mean(filtered_embeddings, axis=0)
            if np.linalg.norm(final_embedding) > 0:
                final_embedding = final_embedding / np.linalg.norm(final_embedding)
            
            # Save embedding
            embedding_path = self.embedding_folder / f"{name}.npy"
            np.save(embedding_path, final_embedding)
            
            # Update database
            self.family_db[name] = final_embedding
            
            logger.info(f"Created profile for {name} using {len(filtered_embeddings)} embeddings from {processed_files} files")
            return True
            
        except Exception as e:
            logger.error(f"Error creating speaker profile for {name}: {e}")
            return False