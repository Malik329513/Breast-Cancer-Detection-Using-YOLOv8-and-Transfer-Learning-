

from my_project.data_loader import load_csv, filter_images_by_description, replace_image_dir
from my_project.data_cleaning import clean_dicom_data, rename_columns, convert_to_category
from my_project.image_processing import open_image, convert_to_grayscale, crop_borders, process_images
from my_project.visualization import show_image, plot_correlation_matrix, plot_histogram
from my_project.model_training import build_vgg16_model, train_model
from my_project.utils import print_dataframe_info, plot_training_history

def main():
    # Load and clean data
    dicom_data = load_csv('data/dicom_info.csv')
    dicom_cleaned_data = clean_dicom_data(dicom_data)
    
    # Process and visualize images
    cropped_images = filter_images_by_description(dicom_data, 'cropped images')
    show_image(open_image(cropped_images.iloc[0]))
    
    # Model training
    model = build_vgg16_model((224, 224, 3))
    history = train_model(model, X_train, y_train, X_test, y_test, datagen)
    plot_training_history(history)

if __name__ == "__main__":
    main()
