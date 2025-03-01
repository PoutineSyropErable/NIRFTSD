#!/bin/bash

# Define variables
PROJECT_DIR="/home/francois/Documents/University (Real)/Semester 9/Comp 400/RealProject/Latex_Report"
AUX_DIR="/home/francois/.texfiles/"
MAIN_TEX="main.tex"
PDF_FILE="${PROJECT_DIR}/$(basename "$MAIN_TEX" .tex).pdf"

# Compile LaTeX document
latexmk -pdf -interaction=nonstopmode -synctex=1 \
	-outdir="$PROJECT_DIR" \
	-auxdir="$AUX_DIR" \
	"$MAIN_TEX"

# Check if compilation was successful and open PDF
if [[ -f "$PDF_FILE" ]]; then
	echo "Compilation successful! Opening PDF..."
	zathura "$PDF_FILE" &
else
	echo "Error: PDF file not found!"
fi
