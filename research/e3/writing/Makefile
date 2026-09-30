# Makefile for DECAF measurement paper
# Compile: pdfLaTeX (acmart class), BibTeX for references

MAIN = main

# Compiling method: latexmk/pdflatex
METHOD = latexmk
LATEXMKOPTS = -pdf -interaction=nonstopmode -file-line-error
PDFLATEXOPTS = -interaction=nonstopmode -file-line-error

SOURCES = $(MAIN).tex references.bib $(wildcard sections/*.tex)

# make deletion work on Windows
ifdef SystemRoot
	RM = del /Q
	OPEN = start
else
	RM = rm -f
	OPEN = open
endif

.PHONY: all view clean distclean

all: $(MAIN).pdf

view: $(MAIN).pdf
	$(OPEN) $(MAIN).pdf

ifeq ($(METHOD),latexmk)

$(MAIN).pdf: $(SOURCES)
	latexmk $(LATEXMKOPTS) $(MAIN)

clean:
	latexmk -c $(MAIN)

else

$(MAIN).pdf: $(SOURCES)
	pdflatex $(PDFLATEXOPTS) $(MAIN)
	bibtex $(MAIN)
	pdflatex $(PDFLATEXOPTS) $(MAIN)
	pdflatex $(PDFLATEXOPTS) $(MAIN)

clean:
	-@$(RM) *.aux *.log *.out *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz

endif

distclean: clean
	-@$(RM) $(MAIN).pdf
