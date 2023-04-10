library(pacman)
p_load(
    tidyverse, data.table, dtplyr, reshape2,
    kableExtra, janitor, readxl, png, Cairo, rbenchmark,
    httr, jsonlite, fedmatch, patchwork,
    corrplot, tidygraph, ggraph, igraph,
    treemap, splitstackshape, stringr, lubridate,
    poweRlaw, voronoiTreemap, ggridges,
    DescTools, stringi, kit, pheatmap
)
# set option
options(dplyr.summarise.inform = FALSE)
options(jupyter.plot_mimetypes = "image/png")
options(repr.plot.width = 8, repr.plot.height = 5)

gray_scale <- c("#F3F4F8", "#D2D4DA",  "#bcc0ca",
                "#D3D3D3", "#B3B5BD", "#838383",
                "#9496A1", "#7d7f89", "#777986",
                "#656673", "#5B5D6B", "#4d505e",
                "#404352", "#2b2d3b", "#2B2B2B", "#282A3A",
                "#1b1c2a", "#191a2b", "#141626", "#101223")
color_scale <- c("#BB4444", "#EE9988", "#FFFFFF",
                            "#77AADD", "#4477AA")
blue_scale <- c("#DBE7F5", "#95BCE3",
                "#699BCB", "#6597C7", "#4879A9",
                "#206BAD", "#11549B")

cor_col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF",
                            "#77AADD", "#4477AA"))

# check working directory
getwd()

peepsample <- function(dt) {
    dt %>%
    .[sample(.N, 5)] %>%
    kable("pipe")
}

peephead <- function(dt, x=5) {
    dt %>%
    head(x) %>%
    kable("pipe")
}

### ------- get to know the data ------- ###
# use a package called Lahman
library(Lahman)
Batting %>% as.data.table %>% str()
Pitching %>% as.data.table %>% str()
People %>% as.data.table %>% str()

# create a subset of the data
career <- Batting %>%
  filter(AB > 0) %>%
  anti_join(Pitching, by = "playerID") %>%
  group_by(playerID) %>%
  summarize(H = sum(H), AB = sum(AB)) %>%
  mutate(average = H / AB)

career <- People %>%
  as_tibble() %>%
  select(playerID, nameFirst, nameLast) %>%
  unite(name, nameFirst, nameLast, sep = " ") %>%
  inner_join(career, by = "playerID") %>%
  select(-playerID)

career %>% str()

career %>% head(5) %>% kable("pipe", digits = 3)


career %>% as.data.table() -> career

career %>%
    .[order(-average)] %>% peephead(5)


# fit with beta distribution
career %>%
    .[AB >= 500] %>%
    with(MASS::fitdistr(average, "beta",
                start = list(shape1 = 1, shape2 = 10))) %>%
    pluck("estimate") %>% round(2)

p_seq = seq(0.1, 0.4, 0.01)

options(repr.plot.width = 9, repr.plot.height = 6)
career %>%
    .[AB >= 500] %>%
    with(hist(average, breaks = 20, freq = FALSE,
              xlab = "Career Batting Average",
    main = "Histogram of Career Batting Average")) %>%
    with(lines(p_seq, dbeta(p_seq, 79, 228), col = "red")) %>%
    with(legend("topright", legend = c("data", "fitted beta"),
                col = c("#242222", "red"), lty = 1, lwd = 2))


# calculate the posterior distribution

career %>%
    .[, posterior := (79 + H) / (79 + 228 + AB)] %>%
    .[order(posterior)] %>%
    head(5) %>% kable("pipe", digits = 3)


career %>%
    ggplot(aes(x = average, y = posterior, color = AB)) +
    geom_point() +
    theme_bw() +
    geom_abline(intercept = 0.259, slope = 0, lty = 2,
                    col = "blue") +
    labs(x = "Batting Average", y = "Posterior Estimated Average",
         title = "Posterior Probability of Baseball Batting Average") +
    scale_color_gradient(low = "#848588", high = "#e03a0c") +
    # hightlight selected observations
    geom_point(data = career[average == 1.0], color = "#238420",
                                        size = 2.5) +
    geom_point(data = career[average == 0.0], color = "#238420",
                                        size = 2.5) 


