# Import core libraries for system operations, FastAPI web server, image processing, MongoDB database access, and TensorFlow Lite ML inference
import os
import io
import time
import json
import subprocess

from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import (
    StreamingResponse,
    HTMLResponse,
    Response,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import pymongo
from bson import ObjectId
from bson.binary import Binary

# import tflite_runtime.interpreter as tflite
from tensorflow.keras.models import load_model

print("[INFO] App starting...")


# Initialize the FastAPI application instance
app = FastAPI()

# Authentication Middleware
@app.middleware("http")
async def require_auth(request: Request, call_next):
    # Exclude certain paths from authentication
    path = request.url.path
    if path == "/login" or path.startswith("/static/") or path == "/favicon.ico":
        return await call_next(request)
    
    # Check for authentication cookie
    auth_token = request.cookies.get("auth")
    if auth_token != "admin":
        return RedirectResponse(url="/login", status_code=303)
        
    response = await call_next(request)
    return response


# Configure static file serving and HTML template rendering directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# Cooldown time to prevent saving snapshots too frequently to the database
SNAP_COOLDOWN_SECONDS = float(os.getenv("SNAP_COOLDOWN_SECONDS", "10.0"))

# Sensor reader writes this JSON regularly
LATEST_SENSOR_JSON = "latest_sensor.json"

# Initialize MongoDB connection and prepare collections for snapshots and sensor data
MONGO_URI = os.getenv("MONGODB_URI")
client = None
db = None
snaps_wssv = None
snaps_healthy = None
sensor_collection = None

if MONGO_URI:
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client["crustascope"]
        # Collection names as requested
        snaps_wssv = db["wssv_snaps"]
        snaps_healthy = db["healthy_snaps"]
        sensor_collection = db["sensor_results"]
        print("[INFO] Connected to MongoDB Atlas.")
    except Exception as e:
        print("[WARN] MongoDB connection failed:", e)
else:
    print("[WARN] MONGODB_URI not set. DB features disabled.")

# Path to the trained model used for shrimp disease detection
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "CrustaScope_model.h5")
model = load_model(MODEL_PATH)
print("[DEBUG] Loading model from:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise Exception("MODEL FILE NOT FOUND")
print("[INFO] H5 Model loaded successfully.")

# Perform ML inference on a camera frame and return prediction confidence
def predict_image(img_bgr: np.ndarray) -> float:
    """
    Run H5 model on a BGR frame and return confidence (float).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224))
    resized = resized.astype(np.float32) / 255.0
    resized = np.expand_dims(resized, axis=0)

    # Predict using Keras model
    output = model.predict(resized, verbose=0)

    # Handle different output shapes safely
    if isinstance(output, list):
        output = output[0]

    conf = float(output[0][0])  # binary classification
    return conf


# Convert model confidence score into a readable shrimp health classification label
def classify_label(conf: float) -> str:
    """
    Convert model confidence into label.
    """
    if conf >= 0.7:
        return "WSSV DETECTED"
    elif conf <= 0.3:
        return "Healthy Shrimp"
    else:
        return "No Shrimp"


# Global runtime state for camera monitoring, latest detection result, and snapshot cooldown tracking
camera = None
monitoring = False
current_camera_index = None

last_result = {
    "label": None,
    "confidence": None,
    "timestamp": None,
    "snapshot_saved": False,
}

last_snap_time = 0.0



# Read the latest water-quality sensor data from the JSON file written by the sensor reader
def read_latest_sensor():
    """
    Read most recent sensor JSON written by sensor_reader.py.
    Returns dict or None.
    """
    if not os.path.exists(LATEST_SENSOR_JSON):
        return None
    try:
        with open(LATEST_SENSOR_JSON, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print("[WARN] Could not read latest_sensor.json:", e)
        return None



# Select the correct MongoDB collection based on snapshot type (WSSV or Healthy)
def get_snap_collection(kind: str):
    """
    Return Mongo collection for 'wssv' or 'healthy'.
    """
    if kind == "wssv":
        return snaps_wssv
    elif kind == "healthy":
        return snaps_healthy
    else:
        return None




# Function to store a detected shrimp snapshot and related sensor data into MongoDB
def save_snapshot(label: str, confidence: float, frame_bgr: np.ndarray):
    """
    Save snapshot to MongoDB (WSSV or Healthy only),
    along with current sensor data, using SNAP_COOLDOWN_SECONDS.
    """

    # Use global variable to track last time a snapshot was stored
    global last_snap_time

    # Get current timestamp to enforce cooldown between DB writes
    now_ts = time.time()

    # Skip saving if cooldown period has not passed
    if now_ts - last_snap_time < SNAP_COOLDOWN_SECONDS:
        # Cooldown active – do not spam DB
        return

    # Ensure MongoDB connection and collections are available
    if client is None or db is None or snaps_wssv is None or snaps_healthy is None:
        print("[WARN] MongoDB not available. Snapshot not saved.")
        return

    # Choose correct MongoDB collection depending on detection label
    if label == "WSSV DETECTED":
        col = snaps_wssv
        kind = "wssv"
    elif label == "Healthy Shrimp":
        col = snaps_healthy
        kind = "healthy"
    else:
        # "No Shrimp" detections are ignored and not stored
        return

    # Encode the camera frame into JPEG format for storage
    ok, buf = cv2.imencode(".jpg", frame_bgr)

    # If encoding fails, skip saving the snapshot
    if not ok:
        print("[WARN] Could not encode frame as JPEG.")
        return

    # Convert encoded image buffer into raw bytes
    img_bytes = buf.tobytes()

    # Read latest sensor readings (temperature, pH, turbidity, TDS)
    sensor_doc = read_latest_sensor()

    # Prepare MongoDB document containing detection metadata and image
    doc = {
        "kind": kind,
        "label": label,
        "confidence": float(confidence),
        "created_at": datetime.utcnow().isoformat(),

        # Store image safely in MongoDB as binary data
        "image_bytes": Binary(img_bytes),

        # Store image format for reference
        "image_format": "jpg",

        # Attach sensor readings captured at the time of detection
        "sensor_at_capture": sensor_doc,

        # Save which camera index detected the shrimp
        "camera_index": current_camera_index,
    }

# Insert snapshot record into MongoDB and update cooldown timestamp
    try:
        col.insert_one(doc)
        last_snap_time = now_ts
        print(f"[INFO] Saved {label} snapshot with sensor data.")
    except Exception as e:
        print("[WARN] Error saving snapshot:", e)



# Generate live camera frames, run ML prediction, and stream MJPEG video to the web client
def gen_frames():
    """
    Generator that yields MJPEG frames with overlayed prediction.
    Runs as long as 'monitoring' is True.
    """
    global camera, monitoring, last_result

    if camera is None:
        print("[WARN] gen_frames called but camera is None.")
        return

    print("[INFO] Starting frame generator loop...")
    while monitoring:
        success, frame = camera.read()
        if not success:
            print("[WARN] Camera read failed.")
            break

	# Run ML prediction on the frame, optionally save snapshot, and update latest detection result
        conf = predict_image(frame)
        label = classify_label(conf)
        now_iso = datetime.now().isoformat()

        snapshot_saved = False
        if label in ("WSSV DETECTED", "Healthy Shrimp"):
            save_snapshot(label, conf, frame)
            snapshot_saved = True

        last_result = {
            "label": label,
            "confidence": conf,
            "timestamp": now_iso,
            "snapshot_saved": snapshot_saved,
        }

	# Overlay prediction label and confidence text on the video frame
        if label == "WSSV DETECTED":
            color = (0, 0, 255)
        elif label == "Healthy Shrimp":
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)

        text = f"{label} ({conf * 100:.1f}%)"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )


	# Encode processed frame as JPEG and stream it as MJPEG to the browser
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


    # Release camera resource when streaming loop stops
    if camera is not None:
        camera.release()
        print("[INFO] Camera released in gen_frames().")




@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """
    Login page required for accessing the application.
    """
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    """
    Handle login form submission.
    """
    if username == "admin" and password == "admin123":
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(key="auth", value="admin", httponly=True)
        return response
    else:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid username or password"
        })


@app.get("/logout")
async def logout():
    """
    Logout the user and redirect to login page.
    """
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("auth")
    return response


@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """
    Dashboard shows live sensor cards + recent activity.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/camera", response_class=HTMLResponse)
async def camera_page(request: Request):
    """
    Camera Live Feed ML detection + camera controls.
    """
    return templates.TemplateResponse("camera.html", {"request": request})


@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    """
    Detection Reports gallery of WSSV + Healthy snaps.
    """
    return templates.TemplateResponse("reports.html", {"request": request})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """
    Settings static org/system settings UI.
    """
    return templates.TemplateResponse("settings.html", {"request": request})


# Legacy route to maintain compatibility by redirecting old gallery URL to reports page
@app.get("/gallery", response_class=HTMLResponse, include_in_schema=False)
async def legacy_gallery(request: Request):
    return templates.TemplateResponse("reports.html", {"request": request})


# Legacy route to keep old water URL working by serving the dashboard page
@app.get("/water", response_class=HTMLResponse, include_in_schema=False)
async def legacy_water(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



# Detect available camera indexes on the system for camera selection
@app.get("/cameras")
async def list_cameras():
    return {"cameras": []}


# API endpoint to start shrimp monitoring using the selected camera
@app.post("/start")
async def start_monitor(payload: dict):
    """
    Camera disabled in cloud deployment
    """
    return {"status": "Camera not available in cloud deployment"}


# Stop camera monitoring and release the camera resource
@app.post("/stop")
async def stop_monitor():
    """
    Stop monitoring & release camera.
    """
    global monitoring, camera, current_camera_index
    monitoring = False
    if camera is not None:
        camera.release()
        print("[INFO] Camera released in /stop.")
        camera = None
    current_camera_index = None
    return {"status": "stopped"}



# Stream live MJPEG video feed from the camera to the browser
@app.get("/video_feed")
async def video_feed():
    raise HTTPException(status_code=400, detail="Camera not supported in cloud")
	

# Return the most recent ML detection result for frontend status updates
@app.get("/status")
async def status():
    """
    Latest ML result (label + confidence + timestamp + snapshot flag).
    """
    return last_result




# Provide latest water quality sensor readings for the dashboard
@app.get("/sensor_live")
async def sensor_live():
    """
    Live sensor JSON for dashboard cards.
    """
    data = read_latest_sensor()
    if not data:
        return {
            "timestamp": None,
            "temperature_c": None,
            "ph": None,
            "turbidity": None,
            "tds": None,
        }

    return {
        "timestamp": data.get("timestamp"),
        "temperature_c": data.get("temperature_c"),
        "ph": data.get("ph"),
        "turbidity": data.get("turbidity"),
        "tds": data.get("tds"),
    }





# API endpoint to fetch shrimp detection snapshots from MongoDB for the reports page
@app.get("/snaps")
async def list_snaps(kind: str):
    """
    List WSSV or Healthy snapshots.
    kind = "wssv" or "healthy"
    """

    # If MongoDB connection is not available, return empty result
    if client is None or db is None:
        return {"items": []}

    # Get the correct collection (wssv_snaps or healthy_snaps)
    col = get_snap_collection(kind)

    # Validate requested snapshot type
    if col is None:
        raise HTTPException(status_code=400, detail="Invalid kind")

    # Retrieve all snapshots sorted by newest first
    docs = col.find().sort("created_at", -1)

    # Prepare response list
    items = []

    # Loop through database documents and extract required fields
    for d in docs:

        # Read sensor data stored at capture time
        sensor = d.get("sensor_at_capture", {}) or {}

        # Build structured response object for frontend
        items.append(
            {
                "id": str(d["_id"]),              # Convert Mongo ObjectId to string
                "label": d.get("label"),          # Detection label
                "confidence": d.get("confidence"),# ML prediction confidence
                "created_at": d.get("created_at"),# Timestamp of detection
                "sensor": {
                    "temperature_c": sensor.get("temperature_c"),
                    "ph": sensor.get("ph"),
                    "turbidity": sensor.get("turbidity"),
                    "tds": sensor.get("tds"),
                },
            }
        )

    # Return formatted snapshot list for reports page
    return {"items": items}

# Delete a specific shrimp detection snapshot from MongoDB using its ID
@app.delete("/snap/{kind}/{snap_id}")
async def delete_snap(kind: str, snap_id: str):
    """
    Delete one snapshot by id.
    """
    if client is None or db is None:
        raise HTTPException(status_code=500, detail="DB not available")

    col = get_snap_collection(kind)
    if col is None:
        raise HTTPException(status_code=400, detail="Invalid kind")

    try:
        oid = ObjectId(snap_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid snap id")

    res = col.delete_one({"_id": oid})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")

    return {"status": "deleted"}

# Retrieve a stored shrimp snapshot image from MongoDB and return it as a JPEG response
# The API locates the snapshot using its MongoDB ObjectId and streams the binary image data
# so the frontend can display the detection image (e.g., thumbnail in the reports page)
@app.get("/snap_image/{kind}/{snap_id}")
async def snap_image(kind: str, snap_id: str):
    """
    Raw JPEG for one snapshot (used in card thumbnail).
    """
    if client is None or db is None:
        raise HTTPException(status_code=500, detail="DB not available")

    col = get_snap_collection(kind)
    if col is None:
        raise HTTPException(status_code=400, detail="Invalid kind")

    try:
        oid = ObjectId(snap_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid snap id")

    doc = col.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    img_bytes = doc.get("image_bytes")
    if not img_bytes:
        raise HTTPException(status_code=500, detail="Image missing")

    return Response(content=img_bytes, media_type="image/jpeg")



# Download a stored shrimp detection snapshot from MongoDB in a user-selected image format
# The image binary is read from the database, optionally converted to JPG or PNG using PIL,
# and returned as a downloadable file so the user can save the detection evidence locally
@app.get("/download/{kind}/{snap_id}")
async def download_snap(kind: str, snap_id: str, fmt: str = "jpg"):
    """
    Download snapshot as JPG or PNG.
    """
    if client is None or db is None:
        raise HTTPException(status_code=500, detail="DB not available")

    col = get_snap_collection(kind)
    if col is None:
        raise HTTPException(status_code=400, detail="Invalid kind")

    try:
        oid = ObjectId(snap_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid snap id")

    doc = col.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    img_bytes = doc.get("image_bytes")
    if not img_bytes:
        raise HTTPException(status_code=500, detail="Image missing")

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    fmt = fmt.lower()
    if fmt not in ("jpg", "jpeg", "png"):
        fmt = "jpg"

    buf = io.BytesIO()
    pil_fmt = "JPEG" if fmt in ("jpg", "jpeg") else "PNG"
    img.save(buf, format=pil_fmt)
    buf.seek(0)

    media_type = "image/jpeg" if pil_fmt == "JPEG" else "image/png"
    filename = f"snapshot_{snap_id}.{fmt}"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }
    return Response(content=buf.read(), media_type=media_type, headers=headers)




# Test endpoint to upload an image manually and run shrimp disease detection using the ML model
# The uploaded image is decoded, passed through the TensorFlow Lite model for classification,
# and the result (label, confidence, and optional snapshot save) is returned to the client
@app.post("/upload_test")
async def upload_test(file: UploadFile = File(...)):
    global last_result

    try:
        contents = await file.read()

        # Use PIL (more robust than cv2)
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Convert to OpenCV format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Run ML
    conf = predict_image(img)
    label = classify_label(conf)
    now_iso = datetime.now().isoformat()

    snapshot_saved = False
    if label in ("WSSV DETECTED", "Healthy Shrimp"):
        save_snapshot(label, conf, img)
        snapshot_saved = True

    sensor_data = read_latest_sensor()

    last_result = {
        "label": label,
        "confidence": conf,
        "timestamp": now_iso,
        "snapshot_saved": snapshot_saved,
    }

    return {
        "label": label,
        "confidence": conf,
        "snapshot_saved": snapshot_saved,
        "sensor_at_capture": sensor_data,
    }

# Start the FastAPI application using the Uvicorn ASGI server when the script is run directly
# This launches the backend API server so the web interface and endpoints become accessible
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
