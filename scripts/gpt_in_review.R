# ------------------------------------------------------------------------------
# Load the required libraries
library(pacman)
p_load(stringr, data.table, magrittr, ggplot2, XML, RCurl, knitr, vtree, SPARQL)

# color palette
gray_scale <- c('#F3F4F8','#D2D4DA', '#B3B5BD', 
                '#9496A1', '#7d7f89', '#777986', 
                '#656673', '#5B5D6B', '#4d505e',
                '#404352', '#2b2d3b', '#282A3A',
                '#1b1c2a', '#191a2b',
                '#141626', '#101223')

ft_palette <- c('#990F3D', '#0D7680', '#0F5499', '#262A33', '#FFF1E5')

ft_contrast <- c('#F83', '#00A0DD', '#C00', '#006F9B', '#F2DFCE', '#FF7FAA',
                 '#00994D', '#593380')

peep_head <- function(dt, n = 5) {
    dt %>%
        head(n) %>%
        kable()
}

peep_sample <- function(dt, n = 5) {
    dt %>%
        .[sample(.N, n)] %>%
        kable()
}

peep_tail <- function(dt, n = 5) {
    dt %>%
        tail(n) %>%
        kable()
}
# ------------------------------------------------------------------------------

dt <- fread('../data/gpt-statistics.csv')

dt %>%
    peep_head()


options(repr.plot.width = 7, repr.plot.height = 4.5, repr.plot.res = 300)
dt %>%
    ggplot(aes(x = total_tokens, y = query_time)) +
    geom_point() +
    geom_smooth(method = 'lm') +
    theme_bw(base_size = 14) +
    labs(title = 'Query Time vs Total Tokens (gpt-3.5-turbo-0125)',
         x = 'Total Tokens',
         y = 'Query Time (s)')


options(repr.plot.width = 9, repr.plot.height = 5, repr.plot.res = 300)
par(bg = 'white', cex.axis = 1.5, cex.lab = 1.5)
layout_matrix <- matrix(c(1,1,2), nrow = 1)
# Set the layout
layout(layout_matrix, widths = c(1.3,1), heights = 1)

# First plot
dt %>%
    with(boxplot(query_time ~ total_tokens, data = ., col = ft_palette[1], 
                 main = 'Query Time vs Total Tokens (gpt-3.5-turbo-0125)',
                 xlab = 'Total Tokens',
                 ylab = 'Query Time (s)'))

# Second plot
dt %>%
    with(boxplot(query_time, data = ., col = ft_palette[1], 
                 main = 'Query Time Distribution',
                 ylab = 'Query Time (s)'))


options(repr.plot.width = 7, repr.plot.height = 4.5, repr.plot.res = 300)
par(bg = 'white', cex.axis = 1, cex.lab = 1)
dt %>%
    with(boxplot(query_cost ~ total_tokens, data = ., col = ft_palette[2], 
                 main = 'Query Cost vs Total Tokens (gpt-3.5-turbo-0125)',
                 xlab = 'Total Tokens',
                 ylab = 'Query Cost ($)'))


dt2 <- fread('../data/gpt-statistics-onto.csv')

dt2 %>% str()

dt2 %>%
    peep_head()


# boxplot for query time
dt %>%
    .[, .(query_time, total_tokens)] %>%
    # random sample with 60 rows
    .[sample(.N, 60)] %>%
    # add a new column for the model called 'task'
    .[, task := 'summarization'] -> dt_summ


dt2 %>%
    .[, .(total_tokens, query_time)] %>%
    .[, task := 'ontology'] -> dt_onto


# combine the two data.tables
rbind(dt_summ, dt_onto) %>%
    .[total_tokens < 4000] %>%
    ggplot(aes(x = total_tokens, y = query_time, color = task)) +
    geom_point(size=2) +
    theme_bw(base_size = 14) +
    labs(title = 'Query Time vs Total Tokens (gpt-3.5-turbo-0125)',
         x = 'Total Tokens',
         y = 'Query Time (s)') +
    # move the legend to the top right and with transparent background
    theme(legend.position = c(0.8, 0.8),
          legend.background = element_rect(fill = 'transparent'))
