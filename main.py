import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for image processing
import sys
import os
import importlib.util

# ✅ Ensure SPADE is accessible
SPADE_PATH = os.path.abspath("SPADE-master")  # Use absolute path
sys.path.insert(0, SPADE_PATH)  # Ensure SPADE is first in sys.path

# ✅ Debugging: Show Python Paths
st.sidebar.write("🔍 Python sys.path:", sys.path)

# ✅ Verify SPADE Directory Exists
if not os.path.exists(SPADE_PATH):
    st.error(f"❌ Error: SPADE directory not found at {SPADE_PATH}")
    st.stop()

# ✅ Manually Import `SPADEGenerator`
generator_path = os.path.join(SPADE_PATH, "models", "networks", "generator.py")

if os.path.exists(generator_path):
    spec = importlib.util.spec_from_file_location("generator", generator_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    # ✅ Assign SPADEGenerator manually
    SPADEGenerator = generator.SPADEGenerator
    st.sidebar.success("✅ SPADEGenerator loaded successfully!")
else:
    st.error(f"❌ Generator file not found at {generator_path}. Check your project structure!")
    st.stop()


# ✅ Define the required 'opt' class with necessary attributes
class SPADEOptions:
    def __init__(self):
        self.ngf = 64  # Number of generator filters
        self.label_nc = 35  # Number of label channels (SPADE expects 35)
        self.use_vae = False  # Whether to use Variational Autoencoder
        self.crop_size = 256  # Default image crop size
        self.aspect_ratio = 1.0  # Aspect ratio of the input image
        self.num_upsampling_layers = 'normal'  # Can be 'normal', 'more', or 'most'
        self.z_dim = 256  # Latent space dimension for VAE (only if use_vae=True)
        self.semantic_nc = 35  # Number of semantic label classes
        self.norm_G = "spectralspadesyncbatch3x3"  # Normalization setting for SPADE


# ✅ Create an instance of SPADEOptions
opt = SPADEOptions()


# ✅ Load SPADE Model with 'opt'
@st.cache_resource
def load_model():
    try:
        model = SPADEGenerator(opt)  # Pass the correctly structured 'opt' object
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


# ✅ Initialize Model
model = load_model()


# ✅ Image Preprocessing Function (Convert RGB to 35-Channel Semantic Map)
def preprocess_image(image):
    # Convert image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    # Resize with OpenCV
    resized_image = cv2.resize(image_cv, (256, 256), interpolation=cv2.INTER_AREA)

    # Convert to grayscale (optional) – Adjust based on your dataset
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Generate a 35-channel one-hot encoding tensor (Fake segmentation map)
    num_classes = 35  # SPADE expects 35 channels
    one_hot = np.zeros((256, 256, num_classes), dtype=np.float32)

    # Assign the grayscale values to one of the 35 channels (for testing)
    for i in range(num_classes):
        one_hot[:, :, i] = (gray_image == (i * (255 // num_classes))).astype(np.float32)

    # Convert NumPy array to PyTorch tensor
    one_hot_tensor = torch.tensor(one_hot).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 35, 256, 256)

    return one_hot_tensor


# ✅ Streamlit UI
st.title("🖼️ Scene Understanding - Object Recognition & Segmentation")
st.write("Upload an image and analyze the scene using deep learning.")

# ✅ Upload Image
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # ✅ Preprocess Image
    input_tensor = preprocess_image(image)  # Now has 35 channels

    # ✅ Perform Scene Segmentation
    with torch.no_grad():
        try:
            output = model(input_tensor)  # Run inference

            # ✅ Post-process the Output
            if output.shape[1] > 1:  # Multi-class output
                output_predictions = torch.argmax(output, dim=1).cpu().numpy()[0]
            else:  # Single-class output
                output_predictions = output.squeeze().cpu().numpy()

        except Exception as e:
            st.error(f"❌ Error during model inference: {e}")
            st.stop()

    # ✅ Apply OpenCV Filters (Edge Detection)
    edges = cv2.Canny((output_predictions * 255).astype(np.uint8), 50, 150)

    # ✅ Display Results
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(image)
    ax[0].set_title("📷 Original Image")
    ax[0].axis("off")

    ax[1].imshow(output_predictions, cmap="jet")
    ax[1].set_title("🎨 Segmented Scene")
    ax[1].axis("off")

    ax[2].imshow(edges, cmap="gray")
    ax[2].set_title("🔍 Edge Detection")
    ax[2].axis("off")

    st.pyplot(fig)
