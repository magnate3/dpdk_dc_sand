This is Markdown just to ensure it doesn't get mixed up in all the rst lying
around in this folder.

One tweak needs to be made because for some reason Sphinx's LaTeX renderer
refuses to put bibliographies in the place where they're requested, they
always just go to the end of the document.

So in order to get them in the right place, I do

```
make latex
```

This renders the LaTeX but doesn't actually build a pdf. Using Ye Olde Text
Editor, you can manually move the bibliography into the "Applicable and
Reference Documents" section. Then do

```
make -C _build/latex
```

and it'll run `pdflatex` an appropriate number of passes to produce a nice
crisp PDF with everything in the right place.
