library(TFBSTools)
library(universalmotif)
library(glue)
library(ggbio)
library(ggpattern)
library(ggplot2)
library(GenomicRanges)
library(regioneR)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

files <- list.files(
  path = c("./motifs_D", "./motifs_P"),
  pattern = "\\.meme$",
  full.names = TRUE
)
files
types <- c("ICM","PWM")
for (t in types){
  for (meme_file in files) {
    print(glue("Processing MEME file: {meme_file} with type: {t}"))
    meme_data <- read_meme(meme_file)
    if (length(meme_data) == 0) {
      print(glue("NO MEME DATA: {meme_file}"))
      next
    }
  
    top_n <- min(5, length(meme_data))
    #meme_data <- trim_motifs(meme_data, min.ic=0.1)
    if (top_n == 1) {
      motif_plot <- tryCatch(
        view_motifs(
          use.type = t,
          meme_data,
          show.positions.once = TRUE,
          show.names = TRUE,
          tryRC = TRUE,
          min.overlap = 50,
          relative_entropy = TRUE
        ),
        error = function(e) {
          view_motifs(
            use.type = t,
            meme_data,
            show.positions.once = TRUE,
            show.names = TRUE,
            tryRC = TRUE,
            min.overlap = 50,
            relative_entropy = FALSE
          )
        }
      )
    } else{
      motif_plot <- tryCatch(
        view_motifs(
          use.type = t,
          meme_data[1:top_n],
          show.positions.once = TRUE,
          show.names = TRUE,
          tryRC = TRUE,
          min.overlap = 50,
          relative_entropy = TRUE
        ),
        error = function(e) {
          view_motifs(
            use.type = t,
            meme_data[1:top_n],
            show.positions.once = TRUE,
            show.names = TRUE,
            tryRC = TRUE,
            min.overlap = 50,
            relative_entropy = FALSE
          )
        }
      )
    }
    output_dir <- dirname(meme_file)
    output_base <- tools::file_path_sans_ext(basename(meme_file))
    motif_plot_file <- file.path(output_dir, glue("{output_base}_{t}.png"))
    ggsave(filename=motif_plot_file, plot=motif_plot, width = 12, height = 6, scale = 1.5, dpi = 300, units= "in")
  }
}





