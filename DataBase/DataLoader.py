import pandas as pd
import os
from PIL import Image

class DataBuilder:
    def __init__(self, folders, image_num, features, target, csv_name, min=0, random=False):
        self.folders = folders
        self.image_num = image_num
        self.features = features
        self.target = target
        self.csv_name = csv_name
        self.min = min
        self.random = random
        
    def add_csv(self):
        df = pd.read_csv(self.csv_name)
        if self.random:
            self.df = df.sample(n=self.image_num, random_state=42) 
        else:
            self.df = df[self.min:self.image_num+self.min]
        self.df = self.df[self.features + ["image_id", self.target]]
        
    def add_images_csv(self):
        image_paths = []
        for image_id in self.df['image_id']:
            image_path = self.find_image(image_id)
            image_paths.append(image_path)
        self.df['image_path'] = image_paths
    
    def find_image(self, image_id):
        for folder in self.folders:
            image_path = os.path.join(folder, image_id + '.jpg')
            if os.path.exists(image_path):
                return image_path
        raise FileNotFoundError(f"Image {image_id} not found in any of the specified folders.")
    
    def build(self):
        self.add_csv()
        self.add_images_csv()
        return self.df
        

if __name__ == "__main__":
    Data = DataBuilder(['ham10000_images_part_1', 'ham10000_images_part_2', 'ham10000_images_part_3'],
                       400, ["age"], "dx_type", 'HAM10000_metadata.csv').build()
    
    print(Data.head())