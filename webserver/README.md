# VITON-HD Web Server

A complete web-based virtual try-on application built with FastAPI and VITON-HD.

## Features

- 🌐 **Web Interface**: Clean, responsive web UI for uploading images
- 🔄 **Real-time Processing**: Upload person and cloth images, get try-on results
- 🎯 **VITON-HD Integration**: Uses GMM + ALIAS models for high-quality results
- 📱 **Mobile Friendly**: Works on desktop and mobile devices
- 🚀 **Easy Setup**: No complex installation scripts needed

## Quick Start

1. **Test your setup**:
   ```bash
   cd webserver
   python test_setup.py
   ```

2. **Install dependencies** (if needed):
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**:
   ```bash
   python start_server.py
   ```
   
   Or directly:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to: http://localhost:8000

## File Structure

```
webserver/
├── app.py                 # FastAPI web server
├── viton_runner.py        # VITON-HD inference pipeline
├── preprocess.py          # Image preprocessing (pose, parsing)
├── start_server.py        # Server startup script
├── test_setup.py          # Setup validation script
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface
└── static/
    ├── uploads/          # Temporary uploaded files
    └── results/          # Generated try-on results
```

## How It Works

1. **Upload**: User uploads person image and clothing image
2. **Preprocessing**: 
   - Extract pose keypoints using OpenPose
   - Generate human parsing maps
   - Create cloth masks
   - Generate person-agnostic representations
3. **Inference**:
   - Segmentation generation (predicts new parsing)
   - Clothes deformation using GMM (Geometric Matching Module)
   - Final try-on synthesis using ALIAS generator
4. **Result**: High-resolution try-on image displayed on webpage

## Requirements

- Python 3.11 (recommended)
- CUDA GPU (recommended for performance)
- VITON-HD checkpoint files in `../checkpoints/`:
  - `seg_final.pth`
  - `gmm_final.pth` 
  - `alias_final.pth`
- OpenPose model files in `../checkpoints/pose/`:
  - `pose_deploy_linevec.prototxt`
  - `pose_iter_584000.caffemodel`

## API Endpoints

- `GET /` - Web interface
- `POST /try-on` - Virtual try-on inference
  - Form data: `person_image` (file), `cloth_image` (file)
  - Returns: JSON with result image path
- `POST /cleanup/{session_id}` - Clean up temporary files

## Troubleshooting

### Common Issues

1. **Missing checkpoints**: Make sure all `.pth` files are in `../checkpoints/`
2. **CUDA out of memory**: Reduce batch size or use CPU (slower)
3. **Import errors**: Install missing dependencies with `pip install -r requirements.txt`
4. **Port in use**: Change port in `app.py` or `start_server.py`

### Performance Tips

- Use CUDA GPU for faster inference
- Ensure images are reasonable size (will be resized to 768x1024)
- Close browser tabs after use to free up temporary files

## Development

The codebase is modular and easy to extend:

- **preprocess.py**: Modify image preprocessing pipeline
- **viton_runner.py**: Adjust VITON-HD inference parameters
- **app.py**: Add new API endpoints or modify server behavior
- **templates/index.html**: Customize the web interface

## Notes

- Images are temporarily stored in `static/uploads/` and `static/results/`
- Automatic cleanup removes temporary files after use
- The preprocessing module includes simplified pose/parsing detection
- For production use, consider adding proper human parsing and pose detection models