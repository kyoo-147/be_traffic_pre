from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image

app = Flask(__name__)

@app.route("/")
def root():
    with open("index.html") as file:
        return file.read()

@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(buf.stream)
    return jsonify(boxes)

@app.route("/count", methods=["POST"])
def count():
    buf = request.files["image_file"]
    congestion_status, boxes = detect_objects_on_image1(buf.stream)
    return jsonify({"congestion_status": congestion_status, "boxes": boxes})

def detect_objects_on_image(buf):
    model = YOLO("best.pt")
    results = model.predict(Image.open(buf))
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([x1, y1, x2, y2, result.names[class_id], prob])
    return output

def detect_objects_on_image1(buf):
    model = YOLO("yolov8n.pt")
    results = model.predict(Image.open(buf))
    result = results[0]
    output = []
    count = 0
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([x1, y1, x2, y2, result.names[class_id], prob])
        count += 1
    
    if count > 20:
        congestion_status = "Cảnh báo: Tắc nghẽn giao thông"
    elif count > 10:
        congestion_status = "Cảnh báo: Có thể tắc nghẽn giao thông"
    else:
        congestion_status = "Tình trạng giao thông ổn định"

    with open('result.txt', 'w') as file:
        for box in result.boxes:
            class_id = box.cls[0].item()
            class_name = result.names[class_id]
            file.write(f'{class_name}\n')
    return congestion_status, output

serve(app, host='0.0.0.0', port=8080)
