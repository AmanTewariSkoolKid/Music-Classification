"""
Utilities for loading and preparing FMA dataset
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import DATA_DIR, GENRE_LABELS


def load_fma_metadata(metadata_path):
    """
    Load FMA metadata CSV file
    
    Args:
        metadata_path: Path to tracks.csv or similar metadata file
        
    Returns:
        pd.DataFrame: Metadata with track information
    """
    # FMA metadata has multi-level columns, skip first row
    df = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
    return df


def prepare_dataset(audio_dir, metadata_path, genre_col='top_genre', test_size=0.2):
    """
    Prepare train/val split from FMA dataset
    
    Args:
        audio_dir: Directory containing audio files (e.g., fma_medium/)
        metadata_path: Path to FMA metadata CSV
        genre_col: Column name containing genre information
        test_size: Fraction of data for validation
        
    Returns:
        tuple: (train_files, train_labels, val_files, val_labels)
    """
    audio_dir = Path(audio_dir)
    
    # Load metadata
    df = load_fma_metadata(metadata_path)
    
    # Filter to only genres in GENRE_LABELS
    # (Adjust this based on actual FMA metadata structure)
    # This is a simplified example - adapt to your metadata format
    
    samples = []
    for track_id in df.index:
        # FMA files are organized as 000/000001.mp3, 001/000002.mp3, etc.
        folder = str(track_id).zfill(6)[:3]
        file_path = audio_dir / folder / f"{str(track_id).zfill(6)}.mp3"
        
        if file_path.exists():
            # Get genre from metadata (adjust column access based on FMA structure)
            try:
                # Example: genre = df.loc[track_id, ('track', 'genre_top')]
                # You'll need to adjust this based on actual metadata structure
                genre = None  # Placeholder - extract from your metadata
                
                if genre in GENRE_LABELS:
                    label_idx = GENRE_LABELS.index(genre)
                    samples.append({
                        'track_id': track_id,
                        'path': str(file_path),
                        'genre': genre,
                        'label_idx': label_idx
                    })
            except KeyError:
                continue
    
    # Convert to DataFrame
    dataset_df = pd.DataFrame(samples)
    
    # Split train/val
    train_df, val_df = train_test_split(
        dataset_df,
        test_size=test_size,
        stratify=dataset_df['label_idx'],
        random_state=42
    )
    
    return (
        train_df['path'].tolist(),
        train_df['label_idx'].tolist(),
        val_df['path'].tolist(),
        val_df['label_idx'].tolist()
    )


def save_split(train_files, train_labels, val_files, val_labels, output_dir=None):
    """
    Save train/val split to CSV files
    
    Args:
        train_files: List of training file paths
        train_labels: List of training labels
        val_files: List of validation file paths
        val_labels: List of validation labels
        output_dir: Directory to save CSV files (default: DATA_DIR)
    """
    if output_dir is None:
        output_dir = DATA_DIR
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Save train split
    train_df = pd.DataFrame({
        'path': train_files,
        'label_idx': train_labels
    })
    train_df.to_csv(output_dir / 'train_split.csv', index=False)
    
    # Save val split
    val_df = pd.DataFrame({
        'path': val_files,
        'label_idx': val_labels
    })
    val_df.to_csv(output_dir / 'val_split.csv', index=False)
    
    print(f"Train set: {len(train_files)} samples")
    print(f"Val set: {len(val_files)} samples")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    print("FMA Dataset Loader")
    print("\nTo prepare your dataset:")
    print("1. Download FMA-medium from https://github.com/mdeff/fma")
    print("2. Extract audio files to data/fma_medium/")
    print("3. Place metadata CSV (tracks.csv) in data/fma_metadata/")
    print("4. Run this script to create train/val splits")
    print("\nNote: You'll need to adapt the metadata parsing code")
    print("      based on the actual FMA metadata structure.")
