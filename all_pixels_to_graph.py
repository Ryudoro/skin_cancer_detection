import os
import cv2
import torch
import torch_geometric
from torch_geometric.data import Data


import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric.utils as pyg_utils


# Convert all pixels of an image to a graph
def load_rgb_image(file_path):
    # Load RGB image using OpenCV
    rgb_image = cv2.imread(file_path)
    # Convert image from BGR to RGB (OpenCV loads images in BGR format by default)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    return rgb_image
    
def convert_to_grayscale(rgb_image):
    # Convert RGB image to grayscale
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    return gray_image

def resize_image(img, new_height, new_width):
    return cv2.resize(img, (new_height, new_width))
    
def rgb_image_to_graph(rgb_image, new_height, new_width):
    # Convert RGB image to grayscale
    gray_image = convert_to_grayscale(rgb_image)

    resize_image(gray_image, new_height, new_width)

    # Initialize lists to store graph data
    edge_index = []
    node_features = []

    # Construct grid graph
    image_height, image_width = gray_image.shape[:2]
    num_nodes = image_height * image_width
    for y in range(image_height):
        for x in range(image_width):
            node_features.append(gray_image[y, x])

            # Add edges to neighboring pixels
            if x > 0:
                edge_index.append([y * image_width + x, y * image_width + (x - 1)])
            if x < image_width - 1:
                edge_index.append([y * image_width + x, y * image_width + (x + 1)])
            if y > 0:
                edge_index.append([y * image_width + x, (y - 1) * image_width + x])
            if y < image_height - 1:
                edge_index.append([y * image_width + x, (y + 1) * image_width + x])

    # Convert lists to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(node_features, dtype=torch.float).view(-1, 1)

    # Create a PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index)

    return data


# Function to visualize the graph
def visualize_graph(data):
    # Convert edge index to NetworkX graph
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    G.add_edges_from(edge_index.T)

    # Plot graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_size=50, font_size=8)
    plt.title('Grid Graph Visualization')
    plt.show()



# TO CHANGE 
new_height = 128
new_width = 128
img_path = "cat_dreaming.jpg"

# Convert RGB image to graph
rgb_img = load_rgb_image(img_path)
data = rgb_image_to_graph(rgb_img, new_height, new_width)
data.num_nodes


# Visualize the graph : TOO LONG
# visualize_graph(graph_data)