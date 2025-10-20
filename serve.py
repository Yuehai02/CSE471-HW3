import os
import base64
import io
import time
from flask import Flask, request, jsonify
from PIL import Image
import torch

os.environ.setdefault("YOLOv5_SKIP_REQUIREMENTS", "1")

app = Flask(__name__)


print("Loading YOLOv5 model...")
model_path = "weights/yolov5s.pt"  
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, trust_repo=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded successfully! (device={device})")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        t0 = time.time()

        data = request.form.get("data")
        if not data:
            return jsonify({"status": 400, "message": "No data received"})
        image_bytes = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        t1 = time.time()

        with torch.inference_mode():
            results = model(image, size=640)
        t2 = time.time()

        detections = results.pandas().xyxy[0].to_dict(orient="records")

        output = []
        for det in detections[:10]:  
            output.append({
                "cls": det["name"],
                "conf": float(det["confidence"]),
                "x1": float(det["xmin"]),
                "y1": float(det["ymin"]),
                "x2": float(det["xmax"]),
                "y2": float(det["ymax"])
            })
        t3 = time.time()

        timing = {
            "pre_ms":   int((t1 - t0) * 1000),  
            "infer_ms": int((t2 - t1) * 1000),  
            "post_ms":  int((t3 - t2) * 1000),  
            "total_ms": int((t3 - t0) * 1000),  
        }
        print(f"[Server Timing] pre={timing['pre_ms']}ms, infer={timing['infer_ms']}ms, "
              f"post={timing['post_ms']}ms, total={timing['total_ms']}ms")

        return jsonify({
            "status": 200,
            "count": len(output),
            "results": output,
            "timing": timing
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"status": 500, "message": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8008))
    app.run(debug=False, host='0.0.0.0', port=port)
