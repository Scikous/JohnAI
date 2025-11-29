from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import uvicorn
from model_utils import load_character
from models import JohnExllamav2

# Global variable to hold the JohnExllamav2 instance
john_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global john_model
    
    # Startup
    print("Starting up...")
    try:
        # Initialize the JohnExllamav2 model instance
        character_info_json = "Sagex/characters/character.json"
        instructions, user_name, character_name = load_character(character_info_json)

        john_model = JohnExllamav2.load_model(
            main_model="turboderp/Qwen2.5-VL-7B-Instruct-exl2",
            tokenizer_model="Qwen/Qwen2.5-VL-7B-Instruct",
            revision="8.0bpw",
            character_name=character_name,
            instructions=instructions,
        )
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {e}")
        john_model = None
    
    yield  # Application runs here
    
    # Shutdown
    print("Shutting down...")
    if john_model:
        john_model.cleanup()
        print("Model cleanup completed")

# Pydantic models for request/response
class DialogueRequest(BaseModel):
    prompt: str
    conversation_history: Optional[List[Dict[str, Any]]] = None
    max_tokens: Optional[int] = 200

class DialogueResponse(BaseModel):
    result: Any  # The response from dialogue_generator (ExLlamaV2DynamicJobAsync object)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Dialogue Generator API",
    description="API endpoint for JohnExllamav2 dialogue generation",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/generate_dialogue", response_model=DialogueResponse)
async def generate_dialogue_endpoint(request: DialogueRequest):
    """
    Generate dialogue using the dialogue_generator function
    
    Args:
        request: DialogueRequest containing prompt, conversation_history, and max_tokens
        
    Returns:
        DialogueResponse containing the result from dialogue_generator
    """
    global john_model
    
    if john_model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
            # We define an async generator function that will be passed to StreamingResponse
        async def stream_generator():
            try:
                # The dialogue_generator returns an async job/generator
                job = await john_model.dialogue_generator(
                    prompt=request.prompt,
                    conversation_history=request.conversation_history,
                    max_tokens=request.max_tokens
                )
                
                # We iterate over the async generator to get tokens as they become available
                async for result in job:
                    result = result.get("text", "")
                    yield result

            except Exception as e:
                # This part is tricky; if an error happens mid-stream, the client
                # might just see a dropped connection. Logging is important.
                print(f"Error during stream generation: {e}")
                # You could potentially yield an error message here if the protocol supports it
                # yield f"ERROR: {e}"

        # Return the stream. The client will receive text chunks as they are yielded.
        return StreamingResponse(stream_generator(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating dialogue: {str(e)}")

@app.post("/cancel_dialogue")
async def cancel_dialogue_endpoint():
    """
    Cancel the currently ongoing dialogue generation
    
    Returns:
        Success message
    """
    global john_model
    
    if john_model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        await john_model.cancel_dialogue_generation()
        return {"message": "Dialogue generation cancelled successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling dialogue: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_initialized": john_model is not None}

if __name__ == "__main__":
    # Run the server

    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        # reload=True
    )