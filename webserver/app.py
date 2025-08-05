from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import uuid
from pathlib import Path
import shutil
from viton_runner import VITONRunner
from preprocess import PreprocessPipeline

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize processors
viton_runner = VITONRunner(
    checkpoint_dir="../checkpoints",
    load_height=1024,
    load_width=768
)
preprocessor = PreprocessPipeline()

# Ensure directories exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/try-on")
async def try_on(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...)
):
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded files
        person_path = f"static/uploads/{session_id}_person.jpg"
        cloth_path = f"static/uploads/{session_id}_cloth.jpg"
        
        with open(person_path, "wb") as buffer:
            shutil.copyfileobj(person_image.file, buffer)
        
        with open(cloth_path, "wb") as buffer:
            shutil.copyfileobj(cloth_image.file, buffer)
        
        # Preprocess images
        processed_data = preprocessor.process_images(person_path, cloth_path, session_id)
        
        # Run VITON-HD inference
        result_path = viton_runner.run_inference(processed_data, session_id)
        
        return JSONResponse({
            "success": True,
            "result_image": f"/static/results/{session_id}_result.jpg",
            "session_id": session_id
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up temporary files for a session"""
    try:
        # Remove upload files
        for pattern in [f"{session_id}_*"]:
            for path in Path("static/uploads").glob(pattern):
                path.unlink(missing_ok=True)
            for path in Path("static/results").glob(pattern):
                path.unlink(missing_ok=True)
        
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)