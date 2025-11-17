"""
Material Design 3 GUI for Music Genre Classification
Uses CustomTkinter for modern, beautiful interface with enhanced UX
"""
import customtkinter as ctk
import torch
from tkinter import filedialog, messagebox
import threading
from pathlib import Path
import json
from datetime import datetime
from typing import Optional
import requests
import sys
from PIL import Image
import subprocess
import os

# Ensure src/ is on sys.path for core modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from inference import load_predictor
from config import MODELS_DIR, GENRE_LABELS, LOGS_DIR

# Material Design 3 Color Palette (Light Mode)
COLORS_LIGHT = {
    "background": "#F8F9FA",       # Softer background
    "primary": "#447D9B",          # Calm blue-gray
    "secondary": "#5C8FA3",        # Muted cyan
    "accent": "#FE7743",           # Coral/orange accent
    "surface": "#FFFFFF",          # White surface
    "surface_variant": "#F0F4F8",  # Light blue-gray
    "text_primary": "#1C1B1F",     # Almost black
    "text_secondary": "#49454F",   # Muted text
    "success": "#4CAF50",          # Green for success
    "error": "#BA1A1A",            # Red for error
    "card": "#FAFBFC",             # Very light card
    "shadow": "#E3E8ED",           # Shadow color for elevation
    "sidebar": "#FFFFFF",          # Sidebar surface
    "hover": "#E8F4F8",            # Hover state
}

# Material Design 3 Color Palette (Dark Mode)
COLORS_DARK = {
    "background": "#1C1B1F",       # Dark background
    "primary": "#89B4CB",          # Lighter blue
    "secondary": "#A8C7D7",        # Light cyan
    "accent": "#FF9471",           # Light coral
    "surface": "#2B2930",          # Dark surface
    "surface_variant": "#36343B",  # Darker variant
    "text_primary": "#E6E1E5",     # Light text
    "text_secondary": "#CAC4D0",   # Muted light text
    "success": "#6EDB74",          # Light green
    "error": "#F2B8B5",            # Light red
    "card": "#36343B",             # Dark card
    "shadow": "#000000",           # Black shadow
    "sidebar": "#2B2930",          # Dark sidebar
    "hover": "#3E3C43",            # Dark hover
}

# Genre icons mapping
GENRE_ICONS = {
    "Electronic": "⚡",
    "Experimental": "🔬",
    "Folk": "🎻",
    "Hip-Hop": "🎤",
    "Instrumental": "🎹",
    "International": "🌍",
    "Pop": "⭐",
    "Rock": "🎸",
    "Classical": "🎼",
    "Country": "🤠",
    "Easy Listening": "☕",
    "Jazz": "🎷",
    "Soul-RnB": "💿",
    "Spoken": "📢",
    "Blues": "🎺",
    "Punk": "🤘"
}

# Current theme
CURRENT_THEME = "light"
COLORS = COLORS_LIGHT.copy()


class GenreClassifierGUI:
    """Main GUI Application for Genre Classification with Material Design 3"""
    
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Music Genre Classifier")
        self.root.geometry("1100x750")  # Wider for sidebar
        self.root.configure(fg_color=COLORS["background"])
        
        # Initialize predictor
        self.predictor: Optional[object] = None
        self.current_file: Optional[str] = None
        self.is_processing = False
        self.dark_mode = False
        self.last_predictions = None
        
        # Setup UI
        self._setup_ui()
        
        # Load model on startup
        self._load_model_async()
    
    def _setup_ui(self):
        """Setup the user interface with sidebar navigation"""
        
        # Header
        self._create_header()
        
        # Main container (sidebar + content)
        main_container = ctk.CTkFrame(
            self.root,
            fg_color=COLORS["background"],
            corner_radius=0
        )
        main_container.pack(fill="both", expand=True)
        
        # Sidebar
        self._create_sidebar(main_container)
        
        # Main content area with padding
        self.main_frame = ctk.CTkFrame(
            main_container,
            fg_color=COLORS["background"],
            corner_radius=0
        )
        self.main_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)
        
        # Upload section
        self._create_upload_section()
        
        # Results section
        self._create_results_section()
        
        # Footer
        self._create_footer()
    
    def _create_header(self):
        """Create header with title and status"""
        header_frame = ctk.CTkFrame(
            self.root,
            fg_color=COLORS["primary"],
            corner_radius=0,
            height=80
        )
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="🎵 Music Genre Classifier",
            font=ctk.CTkFont(family="Segoe UI", size=32, weight="bold"),
            text_color=COLORS["surface"]
        )
        title_label.pack(pady=15)
        
        # Status indicator frame
        self.status_frame = ctk.CTkFrame(
            self.root,
            fg_color=COLORS["secondary"],
            corner_radius=0,
            height=40
        )
        self.status_frame.pack(fill="x")
        self.status_frame.pack_propagate(False)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="⏳ Loading model...",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["surface"]
        )
        self.status_label.pack(pady=8)
    
    def _create_sidebar(self, parent):
        """Create modern sidebar with navigation and settings"""
        sidebar = ctk.CTkFrame(
            parent,
            fg_color=COLORS["sidebar"],
            corner_radius=0,
            width=200,
            border_width=0
        )
        sidebar.pack(side="left", fill="y", padx=0, pady=0)
        sidebar.pack_propagate(False)
        
        # Sidebar title
        sidebar_title = ctk.CTkLabel(
            sidebar,
            text="🎵 Menu",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        sidebar_title.pack(pady=(30, 20), padx=20)
        
        # Separator
        separator1 = ctk.CTkFrame(sidebar, fg_color=COLORS["shadow"], height=2)
        separator1.pack(fill="x", padx=20, pady=10)
        
        # Navigation buttons
        nav_buttons = [
            ("🎯 Classify", self._nav_classify),
            ("📊 About", self._nav_about),
            ("⚙️ Settings", self._nav_settings),
        ]
        
        for text, command in nav_buttons:
            btn = ctk.CTkButton(
                sidebar,
                text=text,
                command=command,
                font=ctk.CTkFont(size=14),
                height=45,
                fg_color="transparent",
                hover_color=COLORS["hover"],
                text_color=COLORS["text_primary"],
                anchor="w",
                corner_radius=10
            )
            btn.pack(fill="x", padx=15, pady=5)
        
        # Spacer
        ctk.CTkFrame(sidebar, fg_color="transparent").pack(expand=True)
        
        # Theme toggle section
        separator2 = ctk.CTkFrame(sidebar, fg_color=COLORS["shadow"], height=2)
        separator2.pack(fill="x", padx=20, pady=10)
        
        theme_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        theme_frame.pack(fill="x", padx=20, pady=15)
        
        theme_label = ctk.CTkLabel(
            theme_frame,
            text="🌓 Theme",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_secondary"]
        )
        theme_label.pack(anchor="w")
        
        self.theme_switch = ctk.CTkSwitch(
            theme_frame,
            text="Dark Mode",
            command=self._toggle_theme,
            font=ctk.CTkFont(size=12),
            progress_color=COLORS["primary"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["secondary"]
        )
        self.theme_switch.pack(anchor="w", pady=(5, 0))
        
        # App version
        version_label = ctk.CTkLabel(
            sidebar,
            text="v1.0.0",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_secondary"]
        )
        version_label.pack(pady=(0, 15))

        # Model metrics (updated when model loads)
        self.model_info_label = ctk.CTkLabel(
            sidebar,
            text="Model: not loaded",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        )
        self.model_info_label.pack(pady=(0, 20), padx=20, anchor="w")
    
    def _nav_classify(self):
        """Navigate to classify page (current page)"""
        pass  # Already on this page
    
    def _nav_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About Music Genre Classifier",
            "🎵 Music Genre Classifier v1.0.0\n\n"
            "A deep learning application for classifying music genres\n"
            "using the FMA dataset.\n\n"
            "Features:\n"
            "• 16 genre classification\n"
            "• CNN-based architecture\n"
            "• Material Design 3 UI\n"
            "• Real-time predictions\n\n"
            "© 2025 Deep Learning Lab Project"
        )
    
    def _nav_settings(self):
        """Show settings dialog"""
        messagebox.showinfo(
            "Settings",
            "⚙️ Settings\n\n"
            "• Toggle dark/light mode using the sidebar switch\n"
            "• Adjust prediction count with the slider\n"
            "• Model: models/best_model.pt\n\n"
            "More settings coming soon!"
        )
    
    def _toggle_theme(self):
        """Toggle between light and dark theme instantly"""
        global COLORS, CURRENT_THEME
        
        self.dark_mode = self.theme_switch.get()
        
        if self.dark_mode:
            COLORS = COLORS_DARK.copy()
            CURRENT_THEME = "dark"
            ctk.set_appearance_mode("dark")
        else:
            COLORS = COLORS_LIGHT.copy()
            CURRENT_THEME = "light"
            ctk.set_appearance_mode("light")
        
        # Apply theme to all elements instantly
        self._apply_theme_colors()
    
    def _apply_theme_colors(self):
        """Apply current theme colors to all UI elements"""
        try:
            # Update root background
            self.root.configure(fg_color=COLORS["background"])
            
            # Update main frame
            self.main_frame.configure(fg_color=COLORS["background"])
            
            # Update status bar colors
            if hasattr(self, 'status_frame'):
                current_status_color = self.status_frame.cget("fg_color")
                # Keep the current status color logic (green/red/secondary)
                if "#4CAF50" in str(current_status_color) or "#6EDB74" in str(current_status_color):
                    self.status_frame.configure(fg_color=COLORS["success"])
                    self.status_label.configure(text_color=COLORS["surface"])
                elif "#BA1A1A" in str(current_status_color) or "#F2B8B5" in str(current_status_color) or "#F44336" in str(current_status_color):
                    self.status_frame.configure(fg_color=COLORS["error"])
                    self.status_label.configure(text_color=COLORS["surface"])
                else:
                    self.status_frame.configure(fg_color=COLORS["secondary"])
                    self.status_label.configure(text_color=COLORS["surface"])
            
            # Update all text colors
            self._update_widget_colors(self.root)
            
        except Exception as e:
            print(f"Theme update error: {e}")
    
    def _update_widget_colors(self, widget):
        """Recursively update colors for all widgets"""
        try:
            widget_type = widget.winfo_class()
            
            # Update based on widget type
            if isinstance(widget, ctk.CTkFrame):
                current_color = widget.cget("fg_color")
                # Map old colors to new theme colors
                if current_color in ["#FFFFFF", "#2B2930"] or "surface" in str(current_color):
                    widget.configure(fg_color=COLORS["surface"])
                elif current_color in ["#F8F9FA", "#1C1B1F"] or "background" in str(current_color):
                    widget.configure(fg_color=COLORS["background"])
                elif current_color in ["#FAFBFC", "#36343B"] or "card" in str(current_color):
                    widget.configure(fg_color=COLORS["card"])
                elif current_color in ["#E3E8ED", "#000000"] or "shadow" in str(current_color):
                    widget.configure(fg_color=COLORS["shadow"])
                elif current_color in ["#F0F4F8", "#36343B"] or "surface_variant" in str(current_color):
                    widget.configure(fg_color=COLORS["surface_variant"])
                    
            elif isinstance(widget, ctk.CTkLabel):
                current_color = widget.cget("text_color")
                if current_color in ["#1C1B1F", "#E6E1E5"] or "text_primary" in str(current_color):
                    widget.configure(text_color=COLORS["text_primary"])
                elif current_color in ["#49454F", "#CAC4D0"] or "text_secondary" in str(current_color):
                    widget.configure(text_color=COLORS["text_secondary"])
                    
            elif isinstance(widget, ctk.CTkButton):
                fg = widget.cget("fg_color")
                if fg in ["#447D9B", "#89B4CB"] or "primary" in str(fg):
                    widget.configure(
                        fg_color=COLORS["primary"],
                        hover_color=COLORS["secondary"]
                    )
                elif fg in ["#FE7743", "#FF9471"] or "accent" in str(fg):
                    widget.configure(
                        fg_color=COLORS["accent"],
                        hover_color=COLORS["primary"]
                    )
                elif fg in ["#5C8FA3", "#A8C7D7"] or "secondary" in str(fg):
                    widget.configure(
                        fg_color=COLORS["secondary"],
                        hover_color=COLORS["primary"]
                    )
                    
            elif isinstance(widget, ctk.CTkEntry):
                widget.configure(
                    fg_color=COLORS["card"],
                    border_color=COLORS["primary"],
                    text_color=COLORS["text_primary"]
                )
                
            elif isinstance(widget, ctk.CTkScrollableFrame):
                widget.configure(fg_color=COLORS["card"])
            
            # Recursively update children
            for child in widget.winfo_children():
                self._update_widget_colors(child)
                
        except Exception as e:
            pass  # Skip widgets that don't support color updates
    
    def _create_upload_section(self):
        """Create file upload section with elevation effect"""
        # Outer shadow frame for elevation effect
        shadow_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color=COLORS["shadow"],
            corner_radius=17
        )
        shadow_frame.pack(fill="x", pady=(0, 20), padx=2)
        
        upload_card = ctk.CTkFrame(
            shadow_frame,
            fg_color=COLORS["surface"],
            corner_radius=15,
            border_width=0
        )
        upload_card.pack(fill="x", padx=2, pady=2)
        
        # Card title with icon
        title_frame = ctk.CTkFrame(upload_card, fg_color="transparent")
        title_frame.pack(fill="x", pady=(20, 10), padx=20)
        
        card_title = ctk.CTkLabel(
            title_frame,
            text="📂 Select Audio File",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        card_title.pack(side="left")
        
        # File selection frame
        file_frame = ctk.CTkFrame(upload_card, fg_color="transparent")
        file_frame.pack(fill="x", padx=20, pady=10)
        
        # File path display
        self.file_entry = ctk.CTkEntry(
            file_frame,
            placeholder_text="No file selected",
            font=ctk.CTkFont(size=14),
            height=45,
            fg_color=COLORS["card"],
            border_color=COLORS["primary"],
            border_width=2
        )
        self.file_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        # Browse button with icon
        self.browse_btn = ctk.CTkButton(
            file_frame,
            text="📁 Browse",
            command=self._browse_file,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45,
            width=130,
            fg_color=COLORS["primary"],
            hover_color=COLORS["secondary"],
            corner_radius=10
        )
        self.browse_btn.pack(side="left")
        
        # Top-K selection
        topk_frame = ctk.CTkFrame(upload_card, fg_color="transparent")
        topk_frame.pack(fill="x", padx=20, pady=10)
        
        topk_label = ctk.CTkLabel(
            topk_frame,
            text="Number of predictions:",
            font=ctk.CTkFont(size=14),
            text_color=COLORS["text_secondary"]
        )
        topk_label.pack(side="left", padx=(0, 15))
        
        self.topk_slider = ctk.CTkSlider(
            topk_frame,
            from_=1,
            to=10,
            number_of_steps=9,
            width=200,
            button_color=COLORS["accent"],
            button_hover_color=COLORS["primary"],
            progress_color=COLORS["primary"]
        )
        self.topk_slider.set(5)
        self.topk_slider.pack(side="left", padx=(0, 15))
        
        self.topk_value_label = ctk.CTkLabel(
            topk_frame,
            text="5",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=COLORS["accent"],
            width=30
        )
        self.topk_value_label.pack(side="left")
        
        # Update label when slider moves
        self.topk_slider.configure(command=self._update_topk_label)
        
        # Button frame for predict and save
        button_frame = ctk.CTkFrame(upload_card, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(10, 20))
        
        # Predict button
        self.predict_btn = ctk.CTkButton(
            button_frame,
            text="🎯 Classify Genre",
            command=self._predict_genre,
            font=ctk.CTkFont(size=18, weight="bold"),
            height=55,
            fg_color=COLORS["accent"],
            hover_color=COLORS["primary"],
            corner_radius=12
        )
        self.predict_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.predict_btn.configure(state="disabled")
        
        # Save results button
        self.save_btn = ctk.CTkButton(
            button_frame,
            text="💾 Save",
            command=self._save_results,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=55,
            width=120,
            fg_color=COLORS["secondary"],
            hover_color=COLORS["primary"],
            corner_radius=12
        )
        self.save_btn.pack(side="left")
        self.save_btn.configure(state="disabled")
    
    def _create_results_section(self):
        """Create results display section with elevation"""
        # Shadow frame for elevation
        shadow_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color=COLORS["shadow"],
            corner_radius=17
        )
        shadow_frame.pack(fill="both", expand=True, padx=2)
        
        results_card = ctk.CTkFrame(
            shadow_frame,
            fg_color=COLORS["surface"],
            corner_radius=15,
            border_width=0
        )
        results_card.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Card title with icon
        title_frame = ctk.CTkFrame(results_card, fg_color="transparent")
        title_frame.pack(fill="x", pady=(20, 10), padx=20)
        
        results_title = ctk.CTkLabel(
            title_frame,
            text="📊 Prediction Results",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        results_title.pack(side="left")
        
        # Scrollable frame for results
        self.results_scroll = ctk.CTkScrollableFrame(
            results_card,
            fg_color=COLORS["card"],
            corner_radius=10
        )
        self.results_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Initial message
        self.initial_msg = ctk.CTkLabel(
            self.results_scroll,
            text="📊 Results will appear here after classification",
            font=ctk.CTkFont(size=16),
            text_color=COLORS["text_secondary"]
        )
        self.initial_msg.pack(pady=50)

        # Placeholder for metrics panel that will appear under results
        self.metrics_panel: Optional[ctk.CTkFrame] = None
    
    def _create_footer(self):
        """Create footer with info"""
        footer_frame = ctk.CTkFrame(
            self.root,
            fg_color=COLORS["secondary"],
            corner_radius=0,
            height=35
        )
        footer_frame.pack(fill="x", side="bottom")
        footer_frame.pack_propagate(False)
        
        footer_label = ctk.CTkLabel(
            footer_frame,
            text="Deep Learning Lab Project | FMA Dataset | 16 Genres",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["surface"]
        )
        footer_label.pack(pady=7)
    
    def _update_topk_label(self, value):
        """Update top-k label when slider changes"""
        self.topk_value_label.configure(text=str(int(value)))
    
    def _load_model_async(self):
        """Load model in background thread"""
        def load():
            try:
                model_path = MODELS_DIR / "best_model.pt"
                if model_path.exists():
                    self.predictor = load_predictor(model_path)
                    self.root.after(0, self._on_model_loaded)
                else:
                    self.root.after(0, self._on_model_not_found)
            except Exception as e:
                self.root.after(0, lambda: self._on_model_error(str(e)))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def _on_model_loaded(self):
        """Called when model is successfully loaded"""
        self.status_label.configure(
            text="✓ Model loaded | Ready to classify",
            text_color=COLORS["success"]
        )
        self.status_frame.configure(fg_color=COLORS["success"])

        # Sidebar label: keep concise to avoid overflow
        if hasattr(self, "model_info_label"):
            self.model_info_label.configure(text="Model: best_model.pt")
        
        # Only enable predict button if file is also selected
        if self.current_file:
            self.predict_btn.configure(state="normal")

        # Render metrics/confusion matrix panel immediately on load
        try:
            self._render_metrics_panel()
        except Exception:
            pass
    
    def _on_model_not_found(self):
        """Called when model file is not found"""
        self.status_label.configure(
            text="⚠ Model not found | Please train a model first",
            text_color=COLORS["error"]
        )
        self.status_frame.configure(fg_color=COLORS["error"])
        if hasattr(self, "model_info_label"):
            self.model_info_label.configure(text="Model: not found")
    
    def _on_model_error(self, error_msg):
        """Called when model loading fails"""
        self.status_label.configure(
            text=f"✗ Error loading model: {error_msg}",
            text_color=COLORS["error"]
        )
        self.status_frame.configure(fg_color=COLORS["error"])
        if hasattr(self, "model_info_label"):
            self.model_info_label.configure(text="Model: error loading")

    def _read_model_metrics(self):
        """Read validation metrics from saved checkpoint if available.

        Returns:
            tuple: (val_acc, val_loss, epoch) if present, else None
        """
        ckpt_path = MODELS_DIR / "best_model.pt"
        if not ckpt_path.exists():
            return None
        # Map to CPU to avoid CUDA dependency for reading
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        val_acc = checkpoint.get("val_acc")
        val_loss = checkpoint.get("val_loss")
        epoch = checkpoint.get("epoch")
        if val_acc is not None and val_loss is not None and epoch is not None:
            # val_acc stored as percent already in training loop; ensure float
            try:
                val_acc = float(val_acc)
                val_loss = float(val_loss)
                epoch = int(epoch)
            except Exception:
                pass
            return val_acc, val_loss, epoch
        return None

    def _read_extended_metrics(self):
        """Read additional metrics (precision/recall/F1) from logs if available.

        Expects a JSON file at LOGS_DIR / 'metrics.json' with keys like
        'precision_macro', 'recall_macro', 'f1_macro'. Returns a dict.
        """
        metrics_path = Path(LOGS_DIR) / "metrics.json"
        data = {}
        try:
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    raw = json.load(f)
                # Normalize keys
                for k in ["precision_macro", "recall_macro", "f1_macro", "accuracy"]:
                    if k in raw and raw[k] is not None:
                        try:
                            data[k] = float(raw[k])
                        except Exception:
                            pass
        except Exception:
            pass
        return data

    def _load_confusion_matrix_image(self):
        """Try to load a confusion matrix image from logs directory.

        Looks for PNG/JPG. Returns a CTkImage or None.
        """
        candidates = [
            Path(LOGS_DIR) / "confusion_matrix.png",
            Path(LOGS_DIR) / "confusion-matrix.png",
            Path(LOGS_DIR) / "confusion_matrix.jpg",
            Path(LOGS_DIR) / "confusion-matrix.jpg",
        ]
        for p in candidates:
            if p.exists():
                try:
                    img = Image.open(p)
                    # Resize to fit card width nicely (~760px inside scroll area)
                    max_w = 760
                    w, h = img.size
                    if w > max_w:
                        ratio = max_w / float(w)
                        img = img.resize((int(w * ratio), int(h * ratio)))
                    return ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                except Exception:
                    return None
        return None
    
    def _browse_file(self):
        """Open file browser to select audio file"""
        filetypes = [
            ("Audio Files", "*.mp3 *.wav *.flac *.ogg *.m4a"),
            ("All Files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes
        )
        
        if filename:
            self.current_file = filename
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, filename)
            
            # Enable predict button if model is loaded and file is selected
            if self.predictor:
                self.predict_btn.configure(state="normal")
    
    def _predict_genre(self):
        """Predict genre for selected file"""
        if not self.current_file:
            messagebox.showwarning("No File", "Please select an audio file first")
            return
        
        if not self.predictor:
            messagebox.showerror("Model Error", "Model is not loaded")
            return
        
        if self.is_processing:
            return
        
        self.is_processing = True
        self.predict_btn.configure(state="disabled", text="⏳ Processing...")
        self.save_btn.configure(state="disabled")
        
        def predict():
            try:
                top_k = int(self.topk_slider.get())
                predictions = self.predictor.predict(self.current_file, top_k=top_k)
                self.last_predictions = predictions  # Store for saving
                self.root.after(0, lambda: self._display_results(predictions))
            except Exception as e:
                self.last_predictions = None
                self.root.after(0, lambda: self._display_error(str(e)))
            finally:
                self.root.after(0, self._reset_predict_button)
        
        thread = threading.Thread(target=predict, daemon=True)
        thread.start()
    
    def _display_results(self, predictions):
        """Display prediction results"""
        # Clear previous results
        for widget in self.results_scroll.winfo_children():
            widget.destroy()
        
        # File info
        file_info = ctk.CTkLabel(
            self.results_scroll,
            text=f"📁 {Path(self.current_file).name}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text_primary"],
            anchor="w"
        )
        file_info.pack(fill="x", pady=(10, 20), padx=10)
        
        # Display each prediction
        for i, pred in enumerate(predictions, 1):
            self._create_prediction_card(i, pred)
        
        # Timestamp
        timestamp = ctk.CTkLabel(
            self.results_scroll,
            text=f"⏱ Classified at {datetime.now().strftime('%H:%M:%S')}",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        )
        timestamp.pack(pady=(20, 10))

        # Render model metrics panel under predictions
        self._render_metrics_panel()
    
    def _create_prediction_card(self, rank, prediction):
        """Create a card for individual prediction with genre icon"""
        genre = prediction['genre']
        probability = prediction['probability']
        genre_icon = GENRE_ICONS.get(genre, "🎵")
        
        # Determine color based on rank with softer gradients
        if rank == 1:
            card_color = COLORS["accent"]
            text_color = COLORS["surface"]
        elif rank <= 3:
            card_color = COLORS["primary"]
            text_color = COLORS["surface"]
        else:
            card_color = COLORS["surface_variant"]
            text_color = COLORS["text_primary"]
        
        # Card frame
        card = ctk.CTkFrame(
            self.results_scroll,
            fg_color=card_color,
            corner_radius=12,
            height=70
        )
        card.pack(fill="x", pady=5, padx=10)
        card.pack_propagate(False)
        
        # Content frame
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Rank, icon and genre
        left_frame = ctk.CTkFrame(content, fg_color="transparent")
        left_frame.pack(side="left", fill="y")
        
        rank_label = ctk.CTkLabel(
            left_frame,
            text=f"#{rank}",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=text_color,
            width=50
        )
        rank_label.pack(side="left", padx=(0, 5))
        
        icon_label = ctk.CTkLabel(
            left_frame,
            text=genre_icon,
            font=ctk.CTkFont(size=24),
            text_color=text_color,
            width=40
        )
        icon_label.pack(side="left", padx=(0, 10))
        
        genre_label = ctk.CTkLabel(
            left_frame,
            text=genre,
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=text_color
        )
        genre_label.pack(side="left")
        
        # Probability
        prob_label = ctk.CTkLabel(
            content,
            text=f"{probability*100:.1f}%",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=text_color
        )
        prob_label.pack(side="right", padx=10)
        
        # Progress bar
        progress_frame = ctk.CTkFrame(content, fg_color="transparent")
        progress_frame.pack(side="right", fill="y", padx=(0, 20))
        
        progress = ctk.CTkProgressBar(
            progress_frame,
            width=150,
            height=20,
            corner_radius=10,
            progress_color=COLORS["success"] if rank == 1 else COLORS["surface"]
        )
        progress.pack(pady=15)
        progress.set(probability)
    
    def _display_error(self, error_msg):
        """Display error message"""
        # Clear previous results
        for widget in self.results_scroll.winfo_children():
            widget.destroy()
        
        error_label = ctk.CTkLabel(
            self.results_scroll,
            text=f"❌ Error: {error_msg}",
            font=ctk.CTkFont(size=16),
            text_color=COLORS["error"],
            wraplength=700
        )
        error_label.pack(pady=50)
        
        messagebox.showerror("Prediction Error", error_msg)
    
    def _reset_predict_button(self):
        """Reset predict button after processing"""
        self.is_processing = False
        self.predict_btn.configure(state="normal", text="🎯 Classify Genre")
        
        # Enable save button if we have predictions
        if self.last_predictions:
            self.save_btn.configure(state="normal")
    
    def _save_results(self):
        """Save prediction results to JSON file"""
        if not self.last_predictions:
            messagebox.showwarning("No Results", "No predictions to save")
            return
        
        # Open save file dialog
        filename = filedialog.asksaveasfilename(
            title="Save Predictions",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if filename:
            try:
                save_data = {
                    "timestamp": datetime.now().isoformat(),
                    "audio_file": self.current_file,
                    "predictions": self.last_predictions,
                    "top_k": len(self.last_predictions)
                }
                
                with open(filename, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Results saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save results:\n{str(e)}")

    def _metric_chip(self, parent, label, value, suffix=""):
        """Small helper to render a metric chip label-value."""
        frame = ctk.CTkFrame(parent, fg_color=COLORS["surface_variant"], corner_radius=8)
        frame.pack(side="left", padx=5, pady=5)
        ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["text_primary"]).pack(padx=8, pady=(6, 0))
        ctk.CTkLabel(frame, text=f"{value}{suffix}", font=ctk.CTkFont(size=12), text_color=COLORS["text_secondary"]).pack(padx=8, pady=(0, 6))
        return frame

    def _render_metrics_panel(self):
        """Render a panel under results showing model metrics and confusion matrix if available."""
        # Destroy previous panel if exists
        if self.metrics_panel is not None and self.metrics_panel.winfo_exists():
            self.metrics_panel.destroy()

        # Create card container
        card = ctk.CTkFrame(self.results_scroll, fg_color=COLORS["surface"], corner_radius=12)
        card.pack(fill="x", padx=10, pady=(10, 20))
        self.metrics_panel = card

        # Header
        header = ctk.CTkLabel(card, text="📈 Model Metrics", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLORS["text_primary"])
        header.pack(anchor="w", padx=15, pady=(12, 6))

        # Chips row
        chips_row = ctk.CTkFrame(card, fg_color="transparent")
        chips_row.pack(fill="x", padx=10, pady=5)

        # Core metrics from checkpoint
        core = self._read_model_metrics()
        if core:
            val_acc, val_loss, epoch = core
            self._metric_chip(chips_row, "Val Acc", f"{val_acc:.2f}", suffix=" %")
            self._metric_chip(chips_row, "Val Loss", f"{val_loss:.4f}")
            self._metric_chip(chips_row, "Best Epoch", f"{epoch+1}")
        else:
            ctk.CTkLabel(chips_row, text="No validation metrics found.", font=ctk.CTkFont(size=12), text_color=COLORS["text_secondary"]).pack(anchor="w", padx=4, pady=6)

        # Extended metrics if available
        ext = self._read_extended_metrics()
        if ext:
            ext_row = ctk.CTkFrame(card, fg_color="transparent")
            ext_row.pack(fill="x", padx=10, pady=(0, 10))
            if "f1_macro" in ext:
                self._metric_chip(ext_row, "F1 (macro)", f"{ext['f1_macro']:.3f}")
            if "precision_macro" in ext:
                self._metric_chip(ext_row, "Precision (macro)", f"{ext['precision_macro']:.3f}")
            if "recall_macro" in ext:
                self._metric_chip(ext_row, "Recall (macro)", f"{ext['recall_macro']:.3f}")

        # Confusion matrix image
        img = self._load_confusion_matrix_image()
        if img is not None:
            img_label = ctk.CTkLabel(card, image=img, text="")
            img_label.image = img  # keep reference
            img_label.pack(padx=10, pady=(10, 12))
        else:
            ctk.CTkLabel(card, text="Confusion matrix not found in logs/.", font=ctk.CTkFont(size=12), text_color=COLORS["text_secondary"]).pack(anchor="w", padx=15, pady=(6, 12))

        # Actions row (Generate / Refresh)
        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.pack(fill="x", padx=10, pady=(0, 12))
        self.refresh_btn = ctk.CTkButton(
            actions,
            text="🔄 Generate / Refresh",
            command=self._on_generate_refresh_clicked,
            height=36,
            fg_color=COLORS["secondary"],
            hover_color=COLORS["primary"],
            corner_radius=8,
            width=180,
        )
        self.refresh_btn.pack(side="left")
        # Optional: open logs folder button
        self.open_logs_btn = ctk.CTkButton(
            actions,
            text="📂 Open Logs Folder",
            command=self._open_logs_folder,
            height=36,
            fg_color=COLORS["surface_variant"],
            hover_color=COLORS["primary"],
            corner_radius=8,
            width=180,
        )
        self.open_logs_btn.pack(side="left", padx=8)

    def _open_logs_folder(self):
        try:
            path = str(LOGS_DIR)
            if os.name == "nt":
                os.startfile(path)
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])
        except Exception:
            pass

    def _on_generate_refresh_clicked(self):
        # Run evaluate.py in a background thread to avoid freezing UI
        def task():
            try:
                self.root.after(0, lambda: self._set_refresh_state(True))
                # Use the same Python interpreter running the GUI
                py = sys.executable
                root = PROJECT_ROOT
                eval_path = root / "src" / "evaluate.py"
                cmd = [py, str(eval_path), "--limit", "300"]
                subprocess.run(cmd, cwd=str(root), check=False)
            finally:
                # Re-render panel (loads new metrics + image)
                self.root.after(0, lambda: self._set_refresh_state(False))
                self.root.after(0, self._render_metrics_panel)

        threading.Thread(target=task, daemon=True).start()

    def _set_refresh_state(self, busy: bool):
        try:
            if busy:
                if hasattr(self, "refresh_btn"):
                    self.refresh_btn.configure(state="disabled", text="⏳ Generating…")
            else:
                if hasattr(self, "refresh_btn"):
                    self.refresh_btn.configure(state="normal", text="🔄 Generate / Refresh")
        except Exception:
            pass
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    app = GenreClassifierGUI()
    app.run()


if __name__ == "__main__":
    main()
