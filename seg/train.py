from ultralytics import YOLO
import os
if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # Load a model
    model = YOLO("yolo11n-seg.pt")

    # Train the model
    train_results = model.train(
        data="dataset/dataset.yaml",
        epochs=100,
        batch = 8,
        imgsz=640,
        device=0,
        workers = 1,
    )

    # # Evaluate model performance on the validation set
    # metrics = model.val()

    # # Perform object detection on an image
    # results = model("c0 (9).jpg")
    # results[0].show()

    # # Export the model to ONNX format
    # path = model.export(format="onnx")  # return path to exported model