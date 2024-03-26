import torch
import torch.nn as nn
import torchvision.models as models
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained CNN model
def load_pretrained_cnn():
    cnn_model = models.resnet18(pretrained=True)
    # Remove the classification layer
    cnn_model = nn.Sequential(*list(cnn_model.children())[:-1])
    # Set the model to evaluation mode
    cnn_model.eval()
    return cnn_model

# Pre-process an image
def preprocess_image(image_path):
    # Load image using your preferred library (e.g., PIL, OpenCV)
    # Perform any necessary pre-processing (e.g., resizing, normalization)
    image = np.random.rand(3, 224, 224)  # Placeholder for pre-processed image

    # Convert the image to a PyTorch tensor and ensure it has the correct data type
    image_tensor = torch.tensor(image, dtype=torch.float32)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

# Extract features from an image using the CNN
def extract_features(cnn_model, image):
    # Pre-process the image
    # Forward pass through the CNN
    with torch.no_grad():
        features = cnn_model(image)
    return features

# Compute edge attributes based on cosine similarity between feature vectors
def compute_edge_attributes(features):
    num_nodes = features.size(1)  # Number of nodes (features)
    edge_attr = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Compute cosine similarity between feature vectors
            similarity = cosine_similarity(features[0, i].reshape(1, -1), features[0, j].reshape(1, -1))[0][0]
            edge_attr.append(similarity)
    return edge_attr

# Construct a graph representation from extracted features and edge attributes
def image_to_graph(features, edge_attr):
    G = nx.Graph()
    num_nodes = features.size(1)  # Number of nodes (features)

    # Add nodes with features
    for i in range(num_nodes):
        node_feature = features[0, i].cpu().numpy()
        G.add_node(i, feature=node_feature)

    # Add edges with attributes
    edge_index = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            G.add_edge(i, j, similarity=edge_attr[edge_index])
            edge_index += 1

    return G

# Visualize the graph
def visualize_graph(graph):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=300, font_size=8, node_color='lightblue', edge_color='gray')
    # Draw edge labels (similarity scores)
    edge_labels = nx.get_edge_attributes(graph, 'similarity')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.title('Graph Representation of Image Features with Edge Attributes')
    plt.show()

# TO CHANGE 
new_height = 128
new_width = 128
img_path = "cat_dreaming.jpg"    

# Load pre-trained CNN
cnn_model = load_pretrained_cnn()

# Pre-process an image
image = preprocess_image(img_path)

# Extract features from the image using the CNN
features = extract_features(cnn_model, image)

# Compute edge attributes based on cosine similarity between feature vectors
edge_attr = compute_edge_attributes(features)

# Construct a graph representation from extracted features and edge attributes
graph = image_to_graph(features, edge_attr)

# Visualize the graph
visualize_graph(graph)