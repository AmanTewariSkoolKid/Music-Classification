# 🎵 Music Genre Classifier - GUI Guide

## 🎨 Beautiful Material Design Interface

Your new GUI application features a stunning Material Design interface with your custom color palette!

### Color Scheme
```
Primary:    #447D9B (Ocean Blue)
Secondary:  #273F4F (Dark Blue-Gray)  
Accent:     #FE7743 (Vibrant Orange)
Background: #D7D7D7 (Light Gray)
Surface:    #FFFFFF (Clean White)
```

---

## 🚀 Quick Start

### 1. Launch the Application
```bash
python run_gui.py
```

### 2. The GUI Layout

```
╔════════════════════════════════════════════════════════════╗
║  🎵 Music Genre Classifier                                 ║
║  (Blue header with white text)                             ║
╠════════════════════════════════════════════════════════════╣
║  Status Bar (Green/Red depending on model status)          ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  ┌─────────────────────────────────────────────────────┐  ║
║  │ Select Audio File                                   │  ║
║  │                                                      │  ║
║  │  [File Path Entry]              [Browse Button]    │  ║
║  │                                                      │  ║
║  │  Number of predictions: ═══○═══ 5                  │  ║
║  │                                                      │  ║
║  │  [🎯 Classify Genre - Orange Button]               │  ║
║  └─────────────────────────────────────────────────────┘  ║
║                                                            ║
║  ┌─────────────────────────────────────────────────────┐  ║
║  │ Prediction Results                                  │  ║
║  │                                                      │  ║
║  │  📁 song.mp3                                        │  ║
║  │                                                      │  ║
║  │  ┌──────────────────────────────────────────────┐  │  ║
║  │  │ #1  Rock              ████████████ 85.3%    │  │  ║
║  │  └──────────────────────────────────────────────┘  │  ║
║  │  ┌──────────────────────────────────────────────┐  │  ║
║  │  │ #2  Pop               ████░░░░░░░ 10.2%     │  │  ║
║  │  └──────────────────────────────────────────────┘  │  ║
║  │  ┌──────────────────────────────────────────────┐  │  ║
║  │  │ #3  Electronic        ██░░░░░░░░░  3.1%     │  │  ║
║  │  └──────────────────────────────────────────────┘  │  ║
║  │                                                      │  ║
║  │  ⏱ Classified at 14:23:45                          │  ║
║  └─────────────────────────────────────────────────────┘  ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
║  Deep Learning Lab Project | FMA Dataset | 16 Genres      ║
╚════════════════════════════════════════════════════════════╝
```

---

## 🎯 Features

### 1. **Header Section** (Blue - #447D9B)
- Large, bold title with music note emoji
- Immediately recognizable and professional

### 2. **Status Bar** (Green ✓ or Red ✗)
- **Green**: Model loaded and ready
- **Red**: Model not found or error
- Real-time status updates

### 3. **File Selection Card** (White surface)
- Clean file path display
- Blue "Browse" button for easy file selection
- Supports: MP3, WAV, FLAC, OGG, M4A

### 4. **Top-K Slider**
- Smooth, interactive slider (1-10 predictions)
- Orange accent color highlights the value
- Real-time value display

### 5. **Classify Button** (Orange - #FE7743)
- Large, prominent button with emoji
- Changes to "Processing..." during classification
- Material Design hover effects

### 6. **Results Display**
- **Scrollable area** for viewing all predictions
- **Ranked cards** with different colors:
  - **#1 (Best)**: Orange background (#FE7743)
  - **#2-3**: Blue background (#447D9B)
  - **#4+**: Light gray background
- **Each card shows**:
  - Rank number (large)
  - Genre name (bold)
  - Confidence percentage
  - Visual progress bar
- File name and timestamp displayed

### 7. **Footer** (Dark - #273F4F)
- Project information
- Professional touch

---

## 🎨 Visual Hierarchy

### Top Prediction (Rank #1)
- **Orange background** - Most eye-catching
- **White text** - High contrast
- Immediately draws attention to the best prediction

### Secondary Predictions (Rank #2-3)
- **Blue background** - Still prominent
- **White text** - Clear readability

### Lower Predictions (Rank #4+)
- **Light gray background** - Subtle
- **Dark text** - Easy to read
- Less emphasis but still accessible

---

## ⚡ User Experience

### Smooth Workflow
1. **Select File** → Browse button opens file dialog
2. **Adjust Predictions** → Slide to choose 1-10 results
3. **Click Classify** → Button changes to "Processing..."
4. **View Results** → Beautiful cards appear with animations
5. **Try Another** → Select new file and repeat

### Threading & Performance
- ✅ Non-blocking UI (runs in background threads)
- ✅ Button disables during processing
- ✅ Visual feedback at every step
- ✅ Error handling with friendly messages

### Accessibility
- 📊 Progress bars for visual feedback
- 🎨 High contrast text and backgrounds
- 📱 Clear, readable fonts (Segoe UI)
- 🔢 Large, touch-friendly buttons

---

## 🛠️ Technical Details

### Built With
- **CustomTkinter** - Modern, customizable tkinter
- **Threading** - Async operations for smooth UI
- **Material Design** - Professional color palette and spacing
- **PyTorch** - Deep learning inference

### Key Components
```python
# Color System
COLORS = {
    "background": "#D7D7D7",
    "primary": "#447D9B",
    "secondary": "#273F4F", 
    "accent": "#FE7743",
    "surface": "#FFFFFF",
}

# Custom Widgets
- CTkFrame (cards and containers)
- CTkButton (Material Design buttons)
- CTkSlider (smooth value selection)
- CTkProgressBar (confidence visualization)
- CTkScrollableFrame (results display)
```

---

## 📸 Visual Elements

### Cards & Shadows
- Rounded corners (12-15px radius)
- Subtle borders (2px)
- Elevated appearance with color contrast

### Typography
- **Headers**: 32px, bold, Segoe UI
- **Titles**: 18px, bold
- **Body**: 14-16px, regular
- **Footer**: 11px, light

### Spacing (Material Design 8px Grid)
- Padding: 20px, 30px
- Margins: 10px, 20px
- Card gaps: 5px between prediction cards

---

## ⚠️ Current Status

**✓ GUI is fully functional and beautiful!**

**Note**: The model needs to be trained first:
```bash
python train.py
```

Once trained, the model at `models/best_model.pt` will be automatically loaded when you launch the GUI.

---

## 🎓 Perfect for Your Project!

This GUI provides:
- ✅ Professional appearance
- ✅ Easy to use
- ✅ Material Design principles
- ✅ Custom color scheme (as requested)
- ✅ Real-time predictions
- ✅ Visual confidence scores
- ✅ Ready for demonstration

**Just train your model and you're ready to present!** 🚀
