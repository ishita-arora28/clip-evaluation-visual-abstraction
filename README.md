# 🧠 CLIP Evaluation on Visually Abstracted Images

This project evaluates the performance of OpenAI’s `CLIP` model (`openai/clip-vit-base-patch32`) on image classification under various levels of visual abstraction using a custom dataset across five image conditions.

## 📁 Dataset

The dataset includes images from **8 categories**:
- `airplane`, `car`, `chair`, `cup`, `dog`, `donkey`, `duck`, `hat`

Each category is tested under **5 different visual conditions**:
- `realistic`, `geons`, `silhouettes`, `blurred`, and `features`

Images are stored in separate folders by condition. The model attempts to match each image to the correct category via zero-shot classification.

## 🧪 Methodology

- Used Hugging Face’s pre-trained CLIP model (`ViT-B/32`) for joint image-text representation.
- For each image, the model evaluates its similarity to each of the 8 textual class labels.
- Classification is based on the label with the highest similarity score.
- Computed:
  - Accuracy for each visual condition
  - Confusion matrix for visual comparison
  - Classification reports (precision, recall, F1)
- Visualized t-SNE embeddings from the vision encoder to observe class-wise separability.

## 🔍 Model Details

- **Architecture**: Vision Transformer (ViT-B/32) + Text Transformer
- **Parameters**:
  - Vision: ~86M
  - Text: ~63M
  - Total: ~150M
- Processes images as 32×32 patches and projects both image and text into a shared 512D space.

## 📊 Results Snapshot

- Accuracy varies by condition, generally highest on realistic images.
- Silhouettes and blurred inputs result in notably lower performance.
- Confusion matrices and classification reports are generated per condition.

## 📦 Tech Stack

- Python, PyTorch, Hugging Face Transformers
- t-SNE (scikit-learn), Matplotlib, Seaborn
- Google Colab

## 📌 How to Run

1. Mount your Google Drive and set `data_root` to the directory containing the image folders.
2. Run all cells to evaluate the model and visualize results.

```python
from google.colab import drive
drive.mount('/content/drive')
data_root = '/content/drive/MyDrive/image_files/image_files/v0'
