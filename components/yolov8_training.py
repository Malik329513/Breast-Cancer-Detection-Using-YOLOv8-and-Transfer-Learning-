# my_project/yolov8_training.py

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

def train_yolov8(model_path, data_yaml, epochs=100, img_size=640, batch_size=16, lr=0.01):
    model = YOLO(model_path)
    model.train(data=data_yaml, epochs=epochs, imgsz=img_size, batch=batch_size, lr0=lr)
    return model

def predict_yolov8(model, image_path):
    image = Image.open(image_path)
    result = model.predict(source=image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    id_to_name = {0: 'BENIGN', 1: 'BENIGN_WITHOUT_CALLBACK', 2: 'MALIGNANT'}
    for i, box in enumerate(result[0].boxes.xyxy):
        box = box.tolist()
        box = [int(coord) for coord in box]
        class_id = int(result[0].boxes.cls[i])
        class_name = id_to_name[class_id]
        confidence = result[0].boxes.conf[i]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{class_name} {confidence:.2f}", fill="white", font=font)

    return image

def save_prediction(image, output_path):
    image.save(output_path)
