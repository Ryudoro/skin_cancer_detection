import os
import shutil
import pandas as pd

images_part_1 = 'ham10000_images_part_1'
images_part_2 = 'ham10000_images_part_2'
metadata_path = 'HAM10000_metadata.csv'

destination_folder = 'classification'

metadata = pd.read_csv(metadata_path)

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for index, row in metadata.iterrows():
    class_folder = os.path.join(destination_folder, row['dx'])
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    image_file = row['image_id'] + '.jpg'

    source_path = None
    if os.path.exists(os.path.join(images_part_1, image_file)):
        source_path = os.path.join(images_part_1, image_file)
    elif os.path.exists(os.path.join(images_part_2, image_file)):
        source_path = os.path.join(images_part_2, image_file)
    
    if source_path is not None:
        shutil.copy(source_path, os.path.join(class_folder, image_file))
    else:
        print(f'Image {image_file} not found in both parts.')

print("Classification of images is complete.")
