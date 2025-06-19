import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import argparse

def remove_silence(audio, sr, top_db=20, frame_length=2048, hop_length=512):
    """
    Remove silence from audio using librosa's trim function
    
    Args:
        audio: Audio signal
        sr: Sample rate
        top_db: Threshold for silence detection (higher = more aggressive)
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
    
    Returns:
        Trimmed audio signal
    """
    # Trim silence from beginning and end
    trimmed_audio, _ = librosa.effects.trim(
        audio, 
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # Remove internal silence gaps (optional - more aggressive)
    # Split audio into segments and remove silent segments
    intervals = librosa.effects.split(
        trimmed_audio, 
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # Concatenate non-silent segments
    if len(intervals) > 0:
        segments = []
        for start, end in intervals:
            segments.append(trimmed_audio[start:end])
        
        # Add small gaps between segments to avoid artifacts
        gap = np.zeros(int(0.1 * sr))  # 0.1 second gap
        final_audio = segments[0]
        for segment in segments[1:]:
            final_audio = np.concatenate([final_audio, gap, segment])
        
        return final_audio
    else:
        return trimmed_audio

def process_audio_file(input_path, output_path, target_sr=16000, silence_threshold=20):
    """
    Process a single audio file: convert to 16kHz and remove silence
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save processed audio
        target_sr: Target sample rate (default 16000 Hz)
        silence_threshold: Silence detection threshold in dB
    """
    try:
        # Load audio file
        print(f"Processing: {input_path}")
        audio, original_sr = librosa.load(input_path, sr=None)
        
        # Resample to target sample rate if needed
        if original_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
            print(f"  Resampled from {original_sr} Hz to {target_sr} Hz")
        
        # Remove silence
        original_duration = len(audio) / target_sr
        trimmed_audio = remove_silence(audio, target_sr, top_db=silence_threshold)
        new_duration = len(trimmed_audio) / target_sr
        
        print(f"  Duration: {original_duration:.2f}s → {new_duration:.2f}s (saved {original_duration-new_duration:.2f}s)")
        
        # Normalize audio to prevent clipping
        if len(trimmed_audio) > 0:
            max_val = np.max(np.abs(trimmed_audio))
            if max_val > 0:
                trimmed_audio = trimmed_audio / max_val * 0.95
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed audio
        sf.write(output_path, trimmed_audio, target_sr)
        print(f"  Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_voice_samples(input_dir="voice_samples", output_dir="output_samples", 
                         target_sr=16000, silence_threshold=20):
    """
    Process all audio files in the voice_samples directory structure
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        target_sr: Target sample rate
        silence_threshold: Silence detection threshold
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Input directory '{input_dir}' not found!")
        return
    
    # Supported audio formats
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    
    processed_count = 0
    failed_count = 0
    
    # Process each person's folder
    for person_folder in input_path.iterdir():
        if person_folder.is_dir():
            person_name = person_folder.name
            print(f"\nProcessing {person_name} folder...")
            
            # Create output folder for this person
            person_output_dir = output_path / person_name
            person_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all audio files in this person's folder
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(person_folder.glob(f"*{ext}"))
            
            for i, audio_file in enumerate(sorted(audio_files), 1):
                # Generate output filename
                output_filename = f"{person_name}_{i}.wav"
                output_file_path = person_output_dir / output_filename
                
                # Process the audio file
                success = process_audio_file(
                    str(audio_file), 
                    str(output_file_path), 
                    target_sr, 
                    silence_threshold
                )
                
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Output saved in: {output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for Alzheimer's patient training")
    parser.add_argument("--input", "-i", default="voice_samples", 
                       help="Input directory (default: voice_samples)")
    parser.add_argument("--output", "-o", default="output_samples", 
                       help="Output directory (default: output_samples)")
    parser.add_argument("--sample-rate", "-sr", type=int, default=16000, 
                       help="Target sample rate in Hz (default: 16000)")
    parser.add_argument("--silence-threshold", "-st", type=int, default=20, 
                       help="Silence detection threshold in dB (default: 20)")
    parser.add_argument("--aggressive", "-a", action="store_true", 
                       help="More aggressive silence removal (threshold=30dB)")
    
    args = parser.parse_args()
    
    # Adjust silence threshold if aggressive mode is enabled
    if args.aggressive:
        args.silence_threshold = 30
        print("Using aggressive silence removal (30dB threshold)")
    
    print("Audio Processing Script for Alzheimer's Training")
    print("=" * 50)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Target sample rate: {args.sample_rate} Hz")
    print(f"Silence threshold: {args.silence_threshold} dB")
    print("=" * 50)
    
    # Process the audio files
    process_voice_samples(
        input_dir=args.input,
        output_dir=args.output,
        target_sr=args.sample_rate,
        silence_threshold=args.silence_threshold
    )