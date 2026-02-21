from ultralytics import YOLO

def main():
    # Load a pretrained classification model
    model = YOLO("yolov8n-cls.pt")  # nano model (fast, good for draft)

    # Train
    model.train(
        data="dataset",        # root folder with train/ and val/
        epochs=25,
        imgsz=224,
        batch=32,
        device=0               # use GPU if available, else set to "cpu"
    )

if __name__ == "__main__":
    main()