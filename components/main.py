# my_project/main.py

from my_project.data_loader import load_csv, filter_images_by_description, replace_image_dir
from my_project.data_cleaning import clean_dicom_data, rename_columns, convert_to_category
from my_project.image_processing import open_image, convert_to_grayscale, crop_borders, process_images, apply_clahe, haze_reduced_local_global
from my_project.visualization import show_image, plot_correlation_matrix, plot_histogram
from my_project.model_training import build_vgg16_model, build_resnet50_model, build_efficientnet_model, train_model
from my_project.yolov8_training import train_yolov8, predict_yolov8, save_prediction
from my_project.utils import print_dataframe_info, plot_training_history

def main():
    # Load and clean data
    dicom_data = load_csv('data/dicom_info.csv')
    dicom_cleaned_data = clean_dicom_data(dicom_data)
    
    # Process and visualize images
    cropped_images = filter_images_by_description(dicom_data, 'cropped images')
    show_image(open_image(cropped_images.iloc[0]))
    
    # Preprocess images
    image = open_image(cropped_images.iloc[0])
    gray_image = convert_to_grayscale(image)
    clahe_image = apply_clahe(np.array(image))
    dehazed_image = haze_reduced_local_global(np.array(image))
    show_image(clahe_image)
    show_image(dehazed_image)

    # Train CNN Models
    model = build_vgg16_model((224, 224, 3))
    history = train_model(model, X_train, y_train, X_test, y_test, datagen)
    plot_training_history(history)

    model = build_resnet50_model((224, 224, 3))
    history = train_model(model, X_train, y_train, X_test, y_test, datagen)
    plot_training_history(history)

    model = build_efficientnet_model((224, 224, 3))
    history = train_model(model, X_train, y_train, X_test, y_test, datagen)
    plot_training_history(history)
    
    # Train YOLOv8 Model
    yolov8_model = train_yolov8('yolov8n.pt', 'data.yaml')
    prediction_image = predict_yolov8(yolov8_model, '/kaggle/input/yolov007/a5/train/images/P_00027_png.rf.44a24a67bcfecc607930dbeceee979b6.jpg')
    save_prediction(prediction_image, '/kaggle/working/output_image.png')

if __name__ == "__main__":
    main()
