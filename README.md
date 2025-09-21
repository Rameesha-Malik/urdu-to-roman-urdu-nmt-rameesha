# Neural Machine Translation: Urdu to Roman Urdu

## BiLSTM Encoder-Decoder with Custom BPE Tokenization

### 🎯 Project Overview

This project implements a state-of-the-art Neural Machine Translation system for converting Urdu text to Roman Urdu transliteration using a BiLSTM encoder-decoder architecture with custom Byte Pair Encoding (BPE) tokenization built from scratch.

### 🏆 Key Achievements

- **BLEU-4 Score**: 0.978 (exceptional performance)
- **Perplexity**: 1.01 (very low, indicating high model confidence)
- **Character Error Rate**: 0.008 (near-perfect character-level accuracy)
- **Custom BPE Tokenizer**: Implemented from scratch without external libraries
- **Comprehensive Experiments**: 4 different configurations tested and analyzed

### 🚀 Live Demo

**Try the model live here**: [Hugging Face Space - Urdu to Roman Urdu Translator](https://huggingface.co/spaces/rameesha146/urdu-to-roman-urdu-nmt-rameesha)

### 📊 Dataset

**Source**: [urdu_ghazals_rekhta dataset](https://github.com/haroonshakeel/urdu_ghazals_rekhta)

- **Training**: 50% of dataset
- **Validation**: 25% of dataset  
- **Testing**: 25% of dataset
- **Domain**: Urdu poetry and literature
- **Preprocessing**: Text normalization, custom BPE tokenization

### 🏗️ Model Architecture

#### BiLSTM Encoder
- **Layers**: 2 bidirectional LSTM layers
- **Hidden Size**: 256 (512 total with bidirectional)
- **Embedding Dimension**: 256
- **Dropout**: 0.5 (best configuration)

#### LSTM Decoder  
- **Layers**: 4 unidirectional LSTM layers
- **Hidden Size**: 512
- **Teacher Forcing**: Applied during training
- **Output**: Roman Urdu character sequences

#### Custom BPE Tokenizer
- **Implementation**: Built from scratch without external libraries
- **Vocabulary Size**: Dynamic based on corpus
- **Subword Units**: Handles out-of-vocabulary words effectively
- **Training**: Learns optimal merge operations from corpus

### 🧪 Experimental Results

| Experiment | Embedding Dim | Hidden Size | Dropout | Learning Rate | BLEU-4 | Perplexity | CER |
|------------|---------------|-------------|---------|---------------|---------|------------|-----|
| Experiment_1_Emb128 | 128 | 512 | 0.3 | 0.001 | 0.978 | 1.02 | 0.009 |
| Experiment_2_Hidden256 | 256 | 256 | 0.3 | 0.001 | 0.868 | 1.08 | 0.058 |
| **Experiment_3_Dropout05** | **256** | **512** | **0.5** | **0.001** | **0.978** | **1.01** | **0.008** |
| Experiment_4_LR0005 | 256 | 512 | 0.3 | 0.0005 | 0.857 | 1.04 | 0.049 |

**Best Model**: Experiment_3_Dropout05 with higher dropout (0.5) providing optimal regularization.

### 📈 Performance Analysis

#### Why Experiment_3 Performed Best:
- **Higher Dropout (0.5)**: Prevented overfitting, improved generalization
- **Optimal Architecture**: 256 embedding dim with 512 hidden size
- **Regularization Balance**: Perfect trade-off between model capacity and generalization

#### Translation Examples:
```
Input:  یہ ایک خوبصورت دن ہے
Output: yeh ek khubsurat din hai

Input:  میں آپ سے محبت کرتا ہوں  
Output: main aap se mohabbat karta hun

Input:  سورج آسمان میں چمک رہا ہے
Output: suraj aasman mein chamak raha hai
```

### 💻 Technical Implementation

#### Framework & Libraries
- **Deep Learning**: PyTorch
- **Web Interface**: Streamlit  
- **Visualization**: Plotly, Matplotlib
- **Data Processing**: Pandas, NumPy
- **Custom Tokenization**: Implemented from scratch

#### Training Configuration
- **Optimizer**: Adam with gradient clipping
- **Loss Function**: Cross-entropy (ignoring padding tokens)
- **Batch Size**: 32
- **Epochs**: 5 (for experiments)
- **Early Stopping**: Based on validation loss

### 🎨 Web Interface Features

The Streamlit application provides:
- **Real-time Translation**: Input Urdu text, get instant Roman Urdu output
- **Interactive Visualizations**: BLEU scores, perplexity, and CER comparisons
- **Model Architecture Display**: Detailed technical specifications
- **Performance Metrics**: Comprehensive results analysis
- **Example Translations**: Predefined high-quality examples

### 📁 Repository Structure

```
├── streamlit_app.py          # Complete Streamlit web application
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

**Note**: Model files and tokenizers are hosted on Hugging Face due to size constraints.

### 🚀 Deployment

#### Local Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/urdu-roman-nmt.git
cd urdu-roman-nmt

# Install dependencies  
pip install -r requirements.txt

# Run locally (requires model files)
streamlit run streamlit_app.py
```

#### Production Deployment
- **Platform**: Hugging Face Spaces
- **URL**: [https://huggingface.co/spaces/rameesha146/urdu-roman](https://huggingface.co/spaces/rameesha146/urdu-to-roman-urdu-nmt-rameesha)
- **Features**: Full model inference, real-time translation
- **Accessibility**: Public, no authentication required

### 📚 Implementation Highlights

#### Custom BPE Tokenizer
```python
class CustomBPETokenizer:
    def train(self, corpus):
        # Learn merge operations from corpus
        self.merges = self.learn_bpe_merges(corpus)
        
    def tokenize(self, text):
        # Apply learned merges for subword tokenization
        return self.apply_bpe(text, self.merges)
```

#### Model Architecture
```python  
# BiLSTM Encoder
encoder = BiLSTMEncoder(
    vocab_size=urdu_vocab_size,
    embedding_dim=256,
    hidden_size=256,  # 512//2 for bidirectional
    num_layers=2,
    dropout=0.5
)

# LSTM Decoder  
decoder = LSTMDecoder(
    vocab_size=roman_vocab_size,
    embedding_dim=256,
    hidden_size=512,
    num_layers=4,
    dropout=0.5
)
```

### 📊 Evaluation Metrics

#### Primary Metrics
- **BLEU-4 Score**: Measures n-gram overlap with reference translations
- **Perplexity**: Indicates model confidence (lower is better)

#### Secondary Metrics  
- **Character Error Rate (CER)**: Character-level accuracy
- **Edit Distance**: Levenshtein distance between prediction and reference

### 🎯 Key Contributions

1. **Custom BPE Implementation**: Built from scratch without external tokenization libraries
2. **Architecture Optimization**: Systematic hyperparameter experimentation
3. **High-Quality Results**: Achieved near-perfect BLEU scores for transliteration task
4. **End-to-End Pipeline**: Complete system from data preprocessing to web deployment
5. **Comprehensive Analysis**: Detailed experimental comparison and performance analysis

### 🔧 Technical Requirements

```
streamlit>=1.28.0
torch>=2.0.0  
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
```

### 📖 Usage

1. **Access Live Demo**: Visit the Hugging Face Space link above
2. **Input Urdu Text**: Enter Urdu text in the input field
3. **Get Translation**: Click translate to get Roman Urdu output
4. **Explore Results**: View experiment comparisons and performance metrics

---

**Built with PyTorch • Custom BPE Tokenization • BiLSTM Encoder-Decoder Architecture**
