"""
Quick launcher for the Music Genre Classifier GUI
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app.gui import main

if __name__ == "__main__":
    print("=" * 60)
    print("  🎵 Music Genre Classifier - GUI Application")
    print("=" * 60)
    print("\nStarting application...")
    print("Note: Model must be trained first (models/best_model.pt)")
    print("\nLoading...\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nApplication closed by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("\nPress Enter to exit...")
