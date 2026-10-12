# TikZ externalization caches figures in build/tikzcache/ (found through TEXINPUTS) and needs shell escape.
use File::Path qw(make_path);
make_path('build/tikzcache');
$pdf_mode = 1;
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode %O %S';
$aux_dir = 'build';
$out_dir = '.';
# Project style files live in style/.
ensure_path('TEXINPUTS', './style//', './build//');
ensure_path('BIBINPUTS', '..', '.');
