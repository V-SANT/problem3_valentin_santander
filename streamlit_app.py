# streamlit_app.py

import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Generator definition (same as in training)
class Generator(nn.Module):
    def __init__(self, noise_dim=100, label_dim=10, img_dim=784):
        super().__init__()
        self.label_emb = nn.Embedding(10, label_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        x = torch.cat((noise, label_input), dim=1)
        return self.model(x).view(-1, 1, 28, 28)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("mnist_generator.pth", map_location=device))
generator.eval()

# Streamlit UI
st.title("🧠 Handwritten Digit Generator")
digit = st.selectbox("Select a digit to generate", list(range(10)))

if st.button("Generate 5 Images"):
    noise_dim = 100
    num_images = 5
    noise = torch.randn(num_images, noise_dim).to(device)
    labels = torch.tensor([digit] * num_images).to(device)

    with torch.no_grad():
        generated_imgs = generator(noise, labels).cpu()

    # Display images
    fig, axes = plt.subplots(1, num_images, figsize=(10, 2))
    for i, img in enumerate(generated_imgs):
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)
