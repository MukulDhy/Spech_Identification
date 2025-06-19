
import asyncio
from speaker_id_enhanced import EnhancedSpeakerIdentifier

async def test_system():
    print("🧪 Testing Speaker Identification System...")
    
    # Initialize system
    identifier = EnhancedSpeakerIdentifier()
    
    # Check if profiles exist
    if not identifier.family_db:
        print("❌ No speaker profiles found!")
        print("Run 'python setup_profiles.py' first")
        return
    
    print(f"✅ Loaded profiles for: {list(identifier.family_db.keys())}")
    
    # Test with a sample audio file
    test_files = [
        "voice_samples/test_audio/test_mom.wav",
        "voice_samples/test_audio/test_dad.wav"
    ]
    
    for test_file in test_files:
        try:
            with open(test_file, 'rb') as f:
                audio_data = f.read()
            
            result = await identifier.identify_speaker(audio_data)
            print(f"\n📁 File: {test_file}")
            print(f"🎯 Identified: {result['speaker']}")
            print(f"🔥 Confidence: {result['confidence']:.2f}")
            print(f"📊 Status: {result['status']}")
            
        except FileNotFoundError:
            print(f"⚠️  Test file not found: {test_file}")
            print("Put test audio files in voice_samples/test_audio/")

if __name__ == "__main__":
    asyncio.run(test_system())