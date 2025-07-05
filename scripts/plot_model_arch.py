from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import os

os.makedirs("outputs/figures", exist_ok=True)

# --- Plot ResNet-4 ---
resnet_path = "models/bird_resnet4_specaug_cosinedecay_20250705_003052.keras"
resnet = load_model(resnet_path)
plot_model(
    resnet,
    to_file="outputs/figures/resnet4_architecture.png",
    show_shapes=True,
    dpi=96
)

# --- Plot CNN Baseline ---
cnn_path = "models/bird_cnn_bn_dropout_20250704_210539.keras"  
cnn = load_model(cnn_path)
plot_model(
    cnn,
    to_file="outputs/figures/cnn_architecture.png",
    show_shapes=True,
    dpi=96
)

print("Architecture diagrams saved to outputs/figures/")
