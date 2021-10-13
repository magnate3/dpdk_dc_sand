# PDF Logger

Goal of the project:
- An automated testing framework that can produce a nice PDF report at the end.

Constraints:
- Use `pytest`.
- Avoid complicated decorators that no one understands.
- Avoid unnecessary esoteric plugins for the test framework. (`nosekatreport`, I
  am looking at you!)
- Use an intermediate step to "cache" results (in this case I used json) in case
  we want to re-render the PDF to fix formatting or something.

Guiding principles:
- Make the tests themselves look as "normal" as possible, without directives
  that are esoteric and difficult to understand.


## Some thoughts
- I've used Marc Welz's original `katdoc.sty` Latex package, adapted slightly
  to make things look a bit more like the new(er) MS Word template.
- My convention with the individual tests has been to put the blurb all in the
  docstring, and then rather than having a separate procedure section, interleave
  the "procedure" with the actual code using `INFO`-level log messages, while
  details are presented using `DEBUG`-level messages.


## Requires
- `texlive-base` at least. I have `texlive-full` installed so I'm not sure what
  packages are required.
- `latexmk`. This was a handy tool if ever there was one.
