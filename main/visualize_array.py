import numpy as np
import tkinter as tk
from tkinter import ttk


def show_numpy_array_table(arr):
    """Displays a NumPy 2D array in a single scrollable table with row headers (Idx) visually separated."""

    if arr.ndim != 2:
        raise ValueError("Only 2D NumPy arrays are supported.")

    rows, cols = arr.shape

    # Create the Tkinter window
    root = tk.Tk()
    root.title("NumPy Array Viewer")

    # Main frame for table and scrollbar
    table_frame = ttk.Frame(root)
    table_frame.pack(fill="both", expand=True)

    # Shared Scrollbars
    scroll_y = ttk.Scrollbar(table_frame, orient="vertical")
    scroll_x = ttk.Scrollbar(table_frame, orient="horizontal")

    # ===========================
    # 1️⃣ Single Treeview: Includes Row Headers (Idx) + Data
    # ===========================
    columns = ["Idx"] + [str(i) for i in range(cols)]  # Add "Idx" for row indices
    tree = ttk.Treeview(table_frame, columns=columns, show="headings")

    # Set column names and widths
    tree.heading("Idx", text="Idx")  # Row index header
    tree.column("Idx", width=60, anchor="center", stretch=False)  # Fixed width for Idx

    for col in columns[1:]:  # Skip "Idx" since we already set it
        tree.heading(col, text=col)  # Column headers
        tree.column(col, width=50, anchor="center")

    # Define Tags for Row Styling (Alternating Colors)
    tree.tag_configure("even_row", background="white")
    tree.tag_configure("odd_row", background="#f0f0f0")  # Light gray for alternate rows

    # Insert row indices + array values into the single table
    for i, row in enumerate(arr):
        tag = "even_row" if i % 2 == 0 else "odd_row"
        idx_with_pipe = f"{i} |"  # Add pipe separator to visually mark the row index
        tree.insert("", "end", values=[idx_with_pipe] + list(row), tags=(tag,))  # Row index as first column

    tree.pack(fill="both", expand=True)

    # Attach scrollbars to the Treeview
    scroll_y.config(command=tree.yview)
    scroll_x.config(command=tree.xview)
    tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    # Pack scrollbars
    scroll_y.pack(side="right", fill="y")
    scroll_x.pack(side="bottom", fill="x")

    # Run the GUI loop
    root.mainloop()


# Example Usage
if __name__ == "__main__":
    array = np.random.randint(0, 100, (64, 32))  # Example 64x32 array
    show_numpy_array_table(array)
