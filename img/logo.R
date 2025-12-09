library(hexSticker)
library(ggplot2)
library(magick)

# 1. Carregar imagem
img <- image_read("img/img_logo.png") 
logo <- image_ggplot(img, interpolate = TRUE)

# 2. Criar o Sticker
s <- sticker(
  subplot = logo,
  package = "",
  p_size = 1,
  s_x = 1,
  s_y = 1,
  s_width = 1.45,
  s_height = 1.45,
  h_fill = "#ffffff",    # Fundo branco
  h_color = "#90ddf9",   # Borda
  h_size = 2.0,          # Tamanho da borda
  url = "https://prdm0.github.io/FastSurvivalSVM",
  u_size = 3.5,
  u_color = "#0F2536",
  white_around_sticker = FALSE, # Mantenha FALSE
  dpi = 300,
  filename = "logo_temp.png" # Nome temporário, ignorar
)

# 3. A CORREÇÃO: Adicionar margem ao redor do plot para a borda não ser cortada
s_fixed <- s + theme(plot.margin = margin(0.2, 0.2, 0.2, 0.2, "cm"))

# 4. Salvar
ggsave(
  filename = "logo.png", 
  plot = s_fixed, 
  width = 43.9, 
  height = 50.8, 
  units = "mm", 
  bg = "transparent", 
  dpi = 300
)