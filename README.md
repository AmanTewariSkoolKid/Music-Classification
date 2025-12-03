
# Music Genre Classification with Deep Learning

A simplified, working implementation for music genre classification using the FMA (Free Music Archive) dataset with a beautiful Material Design GUI.

**REQUIRED: FMA_MEDIUM DATA SET. only tested in FMA_MEDIUM DATASET
[get dataset here.](https://github.com/mdeff/fma "tested only with FMA_medium")**

## Features

- ✅ Audio preprocessing with librosa (log-mel spectrograms)
- ✅ CNN-based genre classifier with PyTorch
- ✅ Training with data augmentation (SpecAugment)
- ✅ **Beautiful Material Design GUI with CustomTkinter**
- ✅ Support for 16 FMA-medium genres
- ✅ Top-k predictions with confidence scores
- ✅ Real-time classification with progress visualization

## Project Structure

```
.
├── config.py           # Configuration settings
├── preprocessing.py    # Audio preprocessing utilities
├── model.py           # CNN model architecture
├── train.py           # Training script
├── inference.py       # Inference utilities
├── gui.py             # Material Design GUI
├── run_gui.py         # GUI launcher
├── requirements.txt   # Python dependencies
├── data/             # Dataset directory
├── models/           # Saved models
├── cache/            # Temporary files
└── logs/             # Training logs
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download FMA Dataset

Download the FMA-medium dataset from: https://github.com/mdeff/fma

Extract to the `data/` directory:

```
data/
└── fma_medium/
    ├── 000/
    ├── 001/
    └── ...
```

### 3. Prepare Dataset

Create CSV files with your train/validation splits:

- `train_files.csv`: Columns = ['path', 'label_idx']
- `val_files.csv`: Columns = ['path', 'label_idx']

Genre indices (0-15):

```
0: Electronic      8: Classical
1: Experimental    9: Country
2: Folk           10: Easy Listening
3: Hip-Hop        11: Jazz
4: Instrumental   12: Soul-RnB
5: International  13: Spoken
6: Pop            14: Blues
7: Rock           15: Punk
```

## Usage

### Running the GUI Application

**Simply run:**

```bash
python run_gui.py
```

Or directly:

```bash
python gui.py
```

The GUI features:

- 🎨 Beautiful Material Design interface with custom color scheme
- 📁 Easy file browser for audio selection
- 🎯 Adjustable top-k predictions (1-10)
- 📊 Visual progress bars and confidence scores
- ⚡ Real-time classification results
- 🎵 Support for MP3, WAV, FLAC, OGG, M4A formats

### Training

```python
from train import train_model
import pandas as pd

# Load your data
train_df = pd.read_csv('train_files.csv')
val_df = pd.read_csv('val_files.csv')

# Train
history = train_model(
    train_files=train_df['path'].tolist(),
    train_labels=train_df['label_idx'].tolist(),
    val_files=val_df['path'].tolist(),
    val_labels=val_df['label_idx'].tolist(),
    save_path='models/best_model.pt'
)
```

### Command-Line Inference (Python)

```python
from inference import load_predictor

# Load model
predictor = load_predictor('models/best_model.pt')

# Predict
results = predictor.predict('path/to/audio.mp3', top_k=5)

for pred in results:
    print(f"{pred['genre']}: {pred['probability']:.2%}")
```

## Model Architecture

- Input: Log-mel spectrograms (128 x 1292)
- 4 Convolutional blocks (32→64→128→256 filters)
- Batch normalization + ReLU + MaxPooling
- Global average pooling
- Fully connected layers with dropout
- Output: 16 genre classes

## GUI Color Scheme

The application uses a custom Material Design palette:

- **Background**: `#D7D7D7` - Light gray
- **Primary**: `#447D9B` - Ocean blue
- **Secondary**: `#273F4F` - Dark blue-gray
- **Accent**: `#FE7743` - Vibrant orange
- **Surface**: `#FFFFFF` - Clean white

## Configuration

Edit `config.py` to customize:

- **Audio parameters**: sample rate, n_fft, hop_length, n_mels
- **Training parameters**: batch size, epochs, learning rate
- **Model architecture**: number of classes

## Performance Tips

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster training
2. **Data Augmentation**: SpecAugment is enabled by default during training
3. **Batch Size**: Adjust based on available memory
4. **Early Stopping**: Monitor validation loss and stop if not improving

## Requirements

- Python 3.8+
- PyTorch 2.1+
- librosa 0.10+
- FastAPI 0.104+
- See `requirements.txt` for full list

## Next Steps

1. Prepare your FMA dataset
2. Create train/val split CSV files
3. Train the model with `train.py`
4. Launch the GUI with `run_gui.py`
5. Start classifying music! 🎵

## Notes

- Audio files are automatically resampled to 22050 Hz
- Input audio is trimmed/padded to 30 seconds
- Supports common audio formats (mp3, wav, flac, ogg, m4a)
- The API uses the model saved at `models/best_model.pt`

## License

Educational project for Deep Learning Lab coursework.
