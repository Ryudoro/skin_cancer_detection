from torchvision import transforms
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import sys,os
sys.path.append(os.getcwd())
from DataBase.DataLoader import DataBuilder

class DataGraphBuilder:
    def __init__(self, data_builder, transform=None):
        self.data_builder = data_builder
        self.df = self.data_builder.build()
        self.transform = transform if transform else self.default_transform()
        
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_images(self):
        image_tensors = []
        for image_path in self.df['image_path']:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img)
            image_tensors.append(img_tensor)
        return torch.stack(image_tensors)
    
    def encode_features(self):
        label_encoders = {}
        for column in ['dx', 'dx_type', 'sex', 'localization']:
            le = LabelEncoder()
            self.df[column] = le.fit_transform(self.df[column])
            label_encoders[column] = le
        
        scaler = StandardScaler()
        self.df['age'] = scaler.fit_transform(self.df[['age']])
        self.label_encoders = label_encoders
    
    def build_graph(self):
        image_features = self.load_images()
        num_patients = len(self.df)
        
        self.encode_features()
        patient_info = torch.tensor(self.df[['age', 'sex', 'localization']].values, dtype=torch.float)
        labels = torch.tensor(self.df[self.data_builder.target].values, dtype=torch.long)
        
        edge_index = self.create_edges(num_patients)
        
        node_features = torch.cat([image_features.view(num_patients, -1), patient_info], dim=1)
        
        return Data(x=node_features, edge_index=edge_index, y=labels)
    
    def create_edges(self, num_patients):
        edge_index = []
        for i in range(num_patients):
            for j in range(i + 1, num_patients):
                if self.df['dx'].iloc[i] == self.df['dx'].iloc[j]:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
        return torch.tensor(edge_index, dtype=torch.long).t()
    
    def visualize_graph_by_dx_type(self, data):
        G = to_networkx(data, to_undirected=True)
        node_colors = self.df[self.data_builder.target].values
        dx_labels = list(self.label_encoders['dx'].classes_)
        cmap = plt.get_cmap('Set1', len(dx_labels))
        norm = plt.Normalize(vmin=0, vmax=len(dx_labels) - 1)
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, seed=42, k=0.8)
        nodes = nx.draw(G, pos, with_labels=False, node_color=node_colors, cmap=cmap, node_size=100, edge_color="gray", alpha=0.7)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(node_colors)
        cbar = plt.colorbar(sm, label="Target (dx)", ax=plt.gca())
        cbar.set_ticks(range(len(dx_labels)))
        cbar.set_ticklabels(dx_labels)
        
        plt.title("Graph Visualization by dx")
        plt.show()
    
    def visualize_graph_by_features(self, data):
        G = to_networkx(data, to_undirected=True)
        node_sizes = (self.df['sex'].values + 1) * 100 
        node_colors = self.df['age'].values
        
        node_sizes = [max(10, size) for size in node_sizes]
        
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, seed=42, k=0.8)
        nx.draw(G, pos, with_labels=False, node_color=node_colors, cmap=plt.get_cmap('coolwarm'), node_size=node_sizes, edge_color="gray", alpha=0.7)
        
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm'), norm=plt.Normalize(vmin=node_colors.min(), vmax=node_colors.max()))
        sm.set_array([])
        plt.colorbar(sm, label="Age", ax=plt.gca())
        plt.title("Graph Visualization by Features (Age and Sex)")
        plt.show()
    
    def visualize_graph_with_images(self, data):
        G = to_networkx(data, to_undirected=True)
        pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(12, 12))
        nx.draw(G, pos, with_labels=False, node_size=100, edge_color="gray", alpha=0.7)
        
        ax = plt.gca()
        for (node, image_path) in zip(G.nodes(), self.df['image_path']):
            img = Image.open(image_path)
            img.thumbnail((25, 25), Image.LANCZOS)
            imagebox = OffsetImage(img, zoom=0.6)
            ab = AnnotationBbox(imagebox, pos[node], frameon=False)
            ax.add_artist(ab)
        
        plt.title("Graph Visualization with Node Images")
        plt.show()
    
def main():
    folders = ['ham10000_images_part_1', 'ham10000_images_part_2', 'ham10000_images_part_3']
    features = ['age', 'sex', 'localization', 'dx_type']
    target = 'dx'
    csv_name = 'HAM10000_metadata.csv'
    
    data_builder = DataBuilder(folders=folders, image_num=10, features=features, target=target, csv_name=csv_name)
    graph_builder = DataGraphBuilder(data_builder)
    

    print("Visualisation des transformations d'images :")
    for i, image_path in enumerate(graph_builder.df['image_path']):
        img = Image.open(image_path).convert('RGB')
        transformed_img = graph_builder.transform(img)
        
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original Image")
        
        plt.subplot(1, 2, 2)
        plt.imshow(transformed_img.permute(1, 2, 0))
        plt.title("Transformed Image")
        
        plt.show()
        
        if i >= 2:
            break
    
    print("\nCaractéristiques des patients avant transformation :")
    print(graph_builder.df.head())
    
    print("\nTransformation et normalisation des caractéristiques des patients...")
    graph_builder.encode_features()
    
    print("\nCaractéristiques des patients après transformation :")
    print(graph_builder.df.head())
    
    print("\nVisualisation des arêtes (edges) du graph :")
    num_patients = len(graph_builder.df)
    edge_index = graph_builder.create_edges(num_patients)
    
    print(f"Nombre de nœuds : {num_patients}")
    print(f"Nombre d'arêtes : {edge_index.shape[1]}")
    print(f"Exemple d'arêtes : {edge_index[:, :10]}")
    
    print("\nVisualisation des caractéristiques des nœuds :")
    data = graph_builder.build_graph()
    print(f"Dimensions des caractéristiques des nœuds : {data.x.shape}")
    print(f"Exemple des caractéristiques des nœuds : \n{data.x[:5]}")

def main2():
    folders = ['ham10000_images_part_1', 'ham10000_images_part_2', 'ham10000_images_part_3']
    features = ['age', 'sex', 'localization', 'dx_type']
    target = 'dx'
    csv_name = 'HAM10000_metadata.csv'
    
    data_builder = DataBuilder(folders=folders, image_num=50, features=features, target=target, csv_name=csv_name, min=50, random = True)
    graph_builder = DataGraphBuilder(data_builder)
    
    data = graph_builder.build_graph()
    

    print("Graph dx :")
    graph_builder.visualize_graph_by_dx_type(data)
    
    print("Graph features :")
    graph_builder.visualize_graph_by_features(data)
    
    print("Graph images :")
    graph_builder.visualize_graph_with_images(data)
    
if __name__ == "__main__":
    main2()
    # main()