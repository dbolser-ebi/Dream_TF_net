label_dir <- '../../data/chipseq_labels'
printf <- function(...) cat(sprintf(...))
fnames <- list.files(label_dir)
for (fname in fnames)
{
  print(fname)
  labels <- read.delim(gzfile(file.path(label_dir, fname)))
  counts <- table(labels[, 4])
  printf('Fraction of bound sites %f\n', counts['B']/(counts['A']+counts['U']))
}
