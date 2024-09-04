

def clean_dicom_data(dicom_data):
    dicom_cleaned_data = dicom_data.copy()
    dicom_cleaned_data.drop(['PatientBirthDate', 'AccessionNumber', 'Columns', 'ContentDate', 'ContentTime',
                             'PatientSex', 'ReferringPhysicianName', 'Rows', 'SOPClassUID', 'SOPInstanceUID',
                             'StudyDate', 'StudyID', 'StudyInstanceUID', 'StudyTime', 'InstanceNumber',
                             'SeriesInstanceUID', 'SeriesNumber'], axis=1, inplace=True)
    dicom_cleaned_data['SeriesDescription'].ffill(inplace=True)
    dicom_cleaned_data['Laterality'].bfill(inplace=True)
    return dicom_cleaned_data

def rename_columns(df, rename_map):
    return df.rename(columns=rename_map)

def convert_to_category(df, columns):
    for col in columns:
        df[col] = df[col].astype('category')
    return df
