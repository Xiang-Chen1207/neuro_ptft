
import os
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import fitz  # PyMuPDF
from io import BytesIO

# Paths
MD_PATH = "/vePFS-0x0d/home/cx/ptft/experiments/feature_pred_validation/feature_class.md"
IMG_DIR = "/vePFS-0x0d/home/cx/ptft/experiments/feature_pred_validation/val_full_final_viz/scatter_plots"
OUTPUT_PDF = "/vePFS-0x0d/home/cx/ptft/experiments/feature_pred_validation/val_full_final_viz/appendix/appendix_scatter_grid.pdf"

# Ensure output dir exists
os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)

def parse_feature_order(md_path):
    features = []
    with open(md_path, 'r') as f:
        lines = f.readlines()
    
    # Regex to find lines like: | 0 | Time | Stats | `mean_abs_amplitude` | ...
    # We want the feature name inside backticks
    for line in lines:
        if line.strip().startswith('|') and 'Index' not in line and '---' not in line:
            parts = line.split('|')
            if len(parts) > 4:
                feat_col = parts[4].strip()
                feat_name = feat_col.replace('`', '').strip()
                if feat_name:
                    features.append(feat_name)
    return features

def create_pdf_grid(features, img_dir, output_path):
    # Layout configuration: 8x8
    COLS = 8
    ROWS = 8
    PLOTS_PER_PAGE = COLS * ROWS
    
    # Page size: A4 Landscape might be better for 8x8 squares?
    # Or just a large square canvas?
    # Let's stick to A4 Portrait but minimize margins.
    # 8.27 x 11.69 inches.
    # 8 cols -> ~1 inch wide per plot.
    FIG_SIZE = (8.27, 11.69)
    
    # We need to assemble existing PDFs.
    # Matplotlib's subplot approach with imshow() only works for raster images (png/jpg).
    # For embedding PDFs into a grid, we need a PDF manipulation library.
    # However, PyMuPDF (fitz) or PyPDF2 are complex to use for "grid layout".
    
    # Alternative: Use matplotlib to layout, but read PDF as image? No, that rasterizes.
    
    # Since we generated "rasterized=True" PDFs in the previous step, 
    # the scatter points are rasterized inside the PDF, but text/lines are vectors.
    # To preserve this structure in the final grid, we must merge them as vector objects.
    
    # Unfortunately, doing this purely in Python without rasterizing the vector parts is tricky 
    # without tools like `pdfjam` or `montage`.
    
    # BUT, since the user wants "clarity", and we already made the scatter points rasterized (high res),
    # the main bottleneck for clarity in the previous PNG grid was likely the downsampling 
    # that happens when matplotlib renders a huge figure into a single page.
    
    # If we stick to the Matplotlib-grid approach, we should use the generated PDFs 
    # but we can't "imshow" a PDF.
    
    # WORKAROUND:
    # Use `pdfjam` (linux tool) if available? 
    # Or, render the PDFs to high-res images just for layouting?
    # NO, that defeats the purpose of "svg/pdf".
    
    # The best way to create a grid of existing PDFs is creating a new PDF
    # and placing the pages of the source PDFs onto the new page canvas.
    # We can use `fitz` (PyMuPDF) for this.
    
    doc = fitz.open()
    
    total_plots = len(features)
    num_pages = (total_plots + PLOTS_PER_PAGE - 1) // PLOTS_PER_PAGE
    
    print(f"Total features: {total_plots}")
    print(f"Generating {num_pages} pages (Layout: {ROWS}x{COLS})...")
    
    current_idx = 0
    
    # Margins and spacing (in points, 1 inch = 72 pts)
    PAGE_WIDTH = 595 # A4
    PAGE_HEIGHT = 842
    MARGIN = 20
    
    # Calculate cell size
    cell_w = (PAGE_WIDTH - 2*MARGIN) / COLS
    cell_h = (PAGE_HEIGHT - 2*MARGIN) / ROWS
    
    # Keep square aspect ratio? Or fit to cell?
    # Plots are square. Let's fit to square.
    # cell_h is usually larger than cell_w in this grid? No, 8x8.
    # 842/8 = 105. 595/8 = 74.
    # So width is the constraint.
    plot_size = min(cell_w, cell_h)
    
    # Spacing to center the grid vertically/horizontally
    grid_w = plot_size * COLS
    grid_h = plot_size * ROWS
    start_x = (PAGE_WIDTH - grid_w) / 2
    start_y = (PAGE_HEIGHT - grid_h) / 2
    
    for page_num in range(num_pages):
        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        
        for r in range(ROWS):
            for c in range(COLS):
                if current_idx >= total_plots:
                    break
                    
                feat_name = features[current_idx]
                pdf_name = feat_name.replace('/', '_').replace(' ', '_') + ".pdf"
                pdf_path = os.path.join(img_dir, pdf_name)
                
                # Calculate position
                # Grid is (r, c). 
                # x = start_x + c * plot_size
                # y = start_y + r * plot_size
                rect = fitz.Rect(
                    start_x + c * plot_size,
                    start_y + r * plot_size,
                    start_x + (c + 1) * plot_size,
                    start_y + (r + 1) * plot_size
                )
                
                if os.path.exists(pdf_path):
                    # Insert PDF page
                    src_doc = fitz.open(pdf_path)
                    page.show_pdf_page(rect, src_doc, 0) # Page 0
                    src_doc.close()
                else:
                    page.insert_text(rect.tl + (10, 20), f"Missing:\n{feat_name}", fontsize=6)
                
                current_idx += 1
                
        print(f"Page {page_num+1} done.")
        
    doc.save(output_path)
    print(f"PDF saved to {output_path}")

def main():
    features = parse_feature_order(MD_PATH)
    if not features:
        print("No features found in markdown.")
        return
        
    create_pdf_grid(features, IMG_DIR, OUTPUT_PDF)

if __name__ == "__main__":
    main()
