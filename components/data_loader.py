import pandas as pd

def load_csv(file_path):
    return pd.read_csv(file_path)

def filter_images_by_description(dicom_data, description):
    return dicom_data[dicom_data['SeriesDescription'] == description].image_path

def replace_image_dir(image_paths, old_dir, new_dir):
    return image_paths.apply(lambda x: x.replace(old_dir, new_dir))
