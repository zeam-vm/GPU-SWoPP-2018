PDF = manuscript.pdf

all: ${PDF}

%.pdf:  %.tex ipsj.cls description.tex ipsjsort.bst reference.bib
	platex $<
	- pbibtex $*
	platex $<
	platex $<
	while ( grep -q '^LaTeX Warning: Label(s) may have changed' $*.log) \
	do platex $<; done
	dvipdfmx $*

description.tex: description.md
	@cat $^ \
	| pandoc -f markdown -t latex -V documentclass=ltjarticle --pdf-engine=lualatex -f markdown-auto_identifiers \
	| sed 's/includegraphics/includegraphics[width=1.0\\columnwidth]/g' \
	| sed 's/\[htbp\]/\[t\]/g' \
	> description.tex

#test.xbb: test.png
#	extractbb test.png

clean:
	@rm -rf description.{dvi,log,tex} manuscript.{pdf,aux,dvi,log,out,blg,bbl} *.xbb
