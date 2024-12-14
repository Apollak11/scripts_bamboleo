from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("plate.pt")

results = model.predict(source = "plate3.jpg",
    show=True,
    line_width=2,
    conf=0.5,
    save=True,
)

plates = []
for idx, result in enumerate(results):
    for detection in result.boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = detection
        plate = {
            "id": idx + 1,
            "type": model.names[int(class_id)],
            "position": {
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
            },
            "length": round((y2 - y1) / 2, 2),
            "width": round((x2 - x1) / 2, 2),
            "confidence": round(confidence, 2),
        }
        plates.append(plate)

diameter = 36 #cm
print(plates[0])