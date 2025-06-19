from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import json
import asyncio
from typing import List
import base64
from speaker_id_enhanced import EnhancedSpeakerIdentifier

from pydantic import BaseModel
import io
from typing import Optional

app = FastAPI(title="Enhanced Speaker Identification API")

# Initialize speaker identifier
speaker_id = EnhancedSpeakerIdentifier()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def get_home():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speaker Identification</title>
    </head>
    <body>
        <h1>Enhanced Speaker Identification System</h1>
        
        <h2>Upload Audio File</h2>
        <form action="/identify" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="audio/*" required>
            <button type="submit">Identify Speaker</button>
        </form>
        
        <h2>Real-time Audio Streaming</h2>
        <button onclick="startRecording()">Start Recording</button>
        <button onclick="stopRecording()">Stop Recording</button>
        <div id="results"></div>
        
        <script>
            let mediaRecorder;
            let websocket;
            let audioChunks = [];
            
            async function startRecording() {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                websocket = new WebSocket("ws://localhost:8000/ws");
                
                websocket.onmessage = function(event) {
                    const result = JSON.parse(event.data);
                    document.getElementById('results').innerHTML = 
                        '<h3>Latest Result:</h3><pre>' + JSON.stringify(result, null, 2) + '</pre>';
                };
                
                mediaRecorder.ondataavailable = async function(event) {
                    if (event.data.size > 0 && websocket.readyState === WebSocket.OPEN) {
                        const audioBlob = event.data;
                        const arrayBuffer = await audioBlob.arrayBuffer();
                        const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
                        websocket.send(JSON.stringify({type: "audio", data: base64}));
                    }
                };
                
                mediaRecorder.start(1000); // Send audio every 1 second
            }
            
            function stopRecording() {
                if (mediaRecorder) {
                    mediaRecorder.stop();
                }
                if (websocket) {
                    websocket.close();
                }
            }
        </script>
    </body>
    </html>
    """)

@app.post("/identify")
async def identify_speaker_upload(file: UploadFile = File(...)):
    """Identify speaker from uploaded audio file"""
    try:
        audio_data = await file.read()
        result = await speaker_id.identify_speaker(audio_data)
        return result
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.post("/create_profile")
async def create_speaker_profile(name: str, files: List[UploadFile] = File(...)):
    """Create a new speaker profile from multiple audio files"""
    try:
        # Save uploaded files temporarily
        temp_files = []
        for file in files:
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            temp_files.append(temp_path)
        
        # Create profile
        success = speaker_id.create_speaker_profile(name, temp_files)
        
        # Clean up
        for temp_file in temp_files:
            Path(temp_file).unlink(missing_ok=True)
        
        if success:
            return {"status": "success", "message": f"Profile created for {name}"}
        else:
            return {"status": "error", "message": f"Failed to create profile for {name}"}
            
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "audio":
                # Decode base64 audio data
                audio_data = base64.b64decode(message["data"])
                
                # Identify speaker
                result = await speaker_id.identify_speaker(audio_data)
                
                # Send result back
                await manager.send_personal_message(json.dumps(result), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/speakers")
async def list_speakers():
    """List all registered speakers"""
    return {"speakers": list(speaker_id.family_db.keys())}

@app.delete("/speaker/{name}")
async def delete_speaker(name: str):
    """Delete a speaker profile"""
    try:
        embedding_path = speaker_id.embedding_folder / f"{name}.npy"
        if embedding_path.exists():
            embedding_path.unlink()
            if name in speaker_id.family_db:
                del speaker_id.family_db[name]
            return {"status": "success", "message": f"Deleted profile for {name}"}
        else:
            return {"status": "error", "message": f"Profile for {name} not found"}
    except Exception as e:
        return {"error": str(e), "status": "error"}
    

class AudioAnalysisRequest(BaseModel):
    audio: str  # Base64 encoded audio data (matching your Node.js 'audio' field)
    format: Optional[str] = "wav"  # Audio format (wav, mp3, etc.)

class AudioAnalysisResponse(BaseModel):
    speaker: str
    confidence: float

@app.post("/analyze-audio", response_model=AudioAnalysisResponse)
async def analyze_audio(request: AudioAnalysisRequest):
    """
    Analyze audio data and identify speaker with confidence
    
    Args:
        request: AudioAnalysisRequest containing base64 encoded audio data
        
    Returns:
        AudioAnalysisResponse with speaker name and confidence percentage
    """
    try:
        # Decode base64 audio data
        try:
            audio_bytes = base64.b64decode(request.audio)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {str(e)}")
        
        # Validate audio data
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio data provided")
            
        # Identify speaker using the existing speaker_id instance
        result = await speaker_id.identify_speaker(audio_bytes)
        
        # Extract speaker and confidence from result
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error occurred"))
            
        speaker_name = result.get("speaker", "Unknown")
        confidence_score = result.get("confidence", 0.0)
        
        # Convert confidence to percentage if it's in decimal format
        if confidence_score <= 1.0:
            confidence_percentage = confidence_score * 100
        else:
            confidence_percentage = confidence_score
            
        return AudioAnalysisResponse(
            speaker=speaker_name,
            confidence=round(confidence_percentage, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

# Alternative endpoint that accepts raw audio file upload
@app.post("/analyze-audio-file", response_model=AudioAnalysisResponse)
async def analyze_audio_file(file: UploadFile = File(...)):
    """
    Analyze uploaded audio file and identify speaker with confidence
    
    Args:
        file: Uploaded audio file
        
    Returns:
        AudioAnalysisResponse with speaker name and confidence percentage
    """
    try:
        # Read file data
        audio_data = await file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file provided")
            
        # Identify speaker
        result = await speaker_id.identify_speaker(audio_data)
        
        # Extract speaker and confidence from result
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error occurred"))
            
        speaker_name = result.get("speaker", "Unknown")
        confidence_score = result.get("confidence", 0.0)
        
        # Convert confidence to percentage if it's in decimal format
        if confidence_score <= 1.0:
            confidence_percentage = confidence_score * 100
        else:
            confidence_percentage = confidence_score
            
        return AudioAnalysisResponse(
            speaker=speaker_name,
            confidence=round(confidence_percentage, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")
PORT = 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)