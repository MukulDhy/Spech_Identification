import os
import glob
from pathlib import Path
from speaker_id_enhanced import EnhancedSpeakerIdentifier
# Set cache to local directory to avoid permission issues
os.environ['HF_HOME'] = './huggingface_cache'
os.environ['SPEECHBRAIN_CACHE_DIR'] = './speechbrain_cache'
def setup_speaker_profiles():
    print("🚀 Setting up Speaker Profiles...")
    
    # Initialize the speaker identifier
    identifier = EnhancedSpeakerIdentifier()
    
    # Get all voice sample folders
    voice_samples_dir = Path("voice_samples")
    
    if not voice_samples_dir.exists():
        print("❌ voice_samples directory not found!")
        print("Please create voice_samples/mom/ and voice_samples/dad/ folders")
        print("And put your audio files there")
        return
    
    # Process each person's folder
    for person_folder in voice_samples_dir.iterdir():
        if person_folder.is_dir() and person_folder.name != "test_audio":
            person_name = person_folder.name
            print(f"\n👤 Processing {person_name}...")
            
            # Get all audio files
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
                audio_files.extend(glob.glob(str(person_folder / ext)))
            
            if not audio_files:
                print(f"⚠️  No audio files found for {person_name}")
                continue
            
            print(f"📁 Found {len(audio_files)} audio files for {person_name}")
            
            # Create speaker profile
            success = identifier.create_speaker_profile(person_name, audio_files)
            
            if success:
                print(f"✅ Successfully created profile for {person_name}")
            else:
                print(f"❌ Failed to create profile for {person_name}")
    
    print(f"\n🎯 Setup Complete!")
    print(f"Created profiles: {list(identifier.family_db.keys())}")
    print(f"Profile files saved in: family_embeddings/")

if __name__ == "__main__":
    setup_speaker_profiles()