# Deep Learning – NPTEL Learning Projects

A curated collection of deep learning models and experiments built from the ground up while following the **NPTEL Deep Learning** course. The repository progresses from foundational neural network concepts to advanced architectures, with each project accompanied by clean, well-commented code and notes.

---

## 📚 About

[NPTEL (National Programme on Technology Enhanced Learning)](https://nptel.ac.in/) offers a comprehensive Deep Learning course that covers both theory and practical implementation. This repository captures that journey — every project here maps to a key topic from the course and is implemented using Python and popular deep learning frameworks.

---

## 🗂️ Topics Covered

| # | Topic | Key Concepts |
|---|-------|-------------|
| 1 | **Introduction to Neural Networks** | Perceptron, activation functions, forward pass |
| 2 | **Backpropagation & Optimization** | Gradient descent, SGD, Adam, learning rate |
| 3 | **Regularization Techniques** | Dropout, batch normalization, L1/L2 |
| 4 | **Convolutional Neural Networks (CNNs)** | Convolution, pooling, feature maps |
| 5 | **Recurrent Neural Networks (RNNs)** | Sequence modeling, BPTT, vanishing gradients |
| 6 | **Long Short-Term Memory (LSTM)** | Gates, cell state, language modeling |
| 7 | **Autoencoders** | Encoder-decoder, latent space, denoising |
| 8 | **Generative Adversarial Networks (GANs)** | Generator, discriminator, training dynamics |
| 9 | **Transfer Learning** | Fine-tuning pretrained models (VGG, ResNet) |
| 10 | **Attention & Transformers** | Self-attention, multi-head attention, BERT basics |

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Frameworks:** PyTorch / TensorFlow / Keras
- **Data handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn, TensorBoard
- **Notebooks:** Jupyter Notebook / Google Colab

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Mithil-7/DL--nptel-learning-projects.git
cd DL--nptel-learning-projects
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run a project
Open the relevant Jupyter notebook for the topic you want to explore:
```bash
jupyter notebook
```
Or open it directly in [Google Colab](https://colab.research.google.com/) by uploading the `.ipynb` file.

---

## 📁 Repository Structure

```
DL--nptel-learning-projects/
│
├── 01_neural_networks/
│   └── perceptron.ipynb
├── 02_backpropagation/
│   └── backprop_scratch.ipynb
├── 03_cnns/
│   └── image_classification.ipynb
├── 04_rnns_lstms/
│   └── text_generation.ipynb
├── 05_autoencoders/
│   └── denoising_autoencoder.ipynb
├── 06_gans/
│   └── dcgan.ipynb
├── 07_transfer_learning/
│   └── fine_tune_resnet.ipynb
├── 08_attention_transformers/
│   └── attention_mechanism.ipynb
├── requirements.txt
└── README.md
```

> **Note:** The folder structure above reflects the planned layout. Projects are added progressively as the course is completed.

---

## 📖 Course Reference

- **Course:** [Deep Learning – NPTEL](https://nptel.ac.in/courses/106106184)
- **Instructor:** Prof. Mitesh M. Khapra, IIT Madras
- **Platform:** NPTEL / SWAYAM

---

## 🤝 Contributing

This is a personal learning repository, but suggestions and improvements are always welcome!

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-topic`)
3. Commit your changes (`git commit -m "Add: your topic"`)
4. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🌟 Acknowledgements

- NPTEL and IIT Madras for making world-class deep learning education freely accessible.
- The open-source community for PyTorch, TensorFlow, and the broader ML ecosystem.
