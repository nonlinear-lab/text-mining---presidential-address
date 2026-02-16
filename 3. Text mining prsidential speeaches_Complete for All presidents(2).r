# ============================================================
# TEXT MINING + TOPIC-SENTIMENT CORRELATION:
# US Presidents' Inaugural Speeches
# ============================================================

# --- Load libraries ---
library(tidyverse)
library(tidytext)
library(tm)
library(wordcloud)
library(SnowballC)
library(ggplot2)
library(textdata)
library(topicmodels)
library(widyr)
library(ggraph)
library(igraph)
library(RColorBrewer)
library(plotly)
library(tidyr)

# ------------------------------------------------------------
# Step 1: Load all speeches automatically
# ------------------------------------------------------------
# Ensure this path is correct for your computer
setwd("C:/R/speech") 

files <- list.files(pattern = "*.txt")

# Check if files exist
if(length(files) == 0) stop("No .txt files found in the directory!")

speeches <- map_df(files, function(f) {
  text <- readLines(f, encoding = "UTF-8", warn = FALSE)
  tibble(president = tools::file_path_sans_ext(f),
         text = paste(text, collapse = " "))
})

# ------------------------------------------------------------
# Step 2: Tokenize and clean
# ------------------------------------------------------------
data("stop_words")

tidy_speeches <- speeches %>%
  unnest_tokens(word, text) %>%
  mutate(word = str_replace_all(word, "[^a-zA-Z]", "")) %>%
  filter(word != "", nchar(word) > 2) %>%
  anti_join(stop_words, by = "word")

# ------------------------------------------------------------
# Step 3: Word Frequency and Wordcloud
# ------------------------------------------------------------
word_freq <- tidy_speeches %>%
  count(word, sort = TRUE) # Count total words for the cloud

set.seed(123)
wordcloud(words = word_freq$word,
          freq = word_freq$n,
          max.words = 100,
          random.order = FALSE,
          colors = brewer.pal(8, "Dark2"))

# ------------------------------------------------------------
# Step 4: Topic Modeling (LDA)
# ------------------------------------------------------------
dtm <- tidy_speeches %>%
  count(president, word) %>%
  cast_dtm(document = president, term = word, value = n)

# k (number of topics) must be less than or equal to number of documents
n_docs <- nrow(as.matrix(dtm))
lda_k <- min(5, n_docs) 

set.seed(123)
lda <- LDA(dtm, k = lda_k, method = "VEM")

topics_tidy <- tidy(lda, matrix = "beta")
gamma_tidy <- tidy(lda, matrix = "gamma")

top_terms <- topics_tidy %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup()

p_topic <- ggplot(top_terms, aes(x = reorder_within(term, beta, topic),
                                 y = beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free_y") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Top Terms per Topic (LDA)", x = "Term", y = "Probability") +
  theme_minimal()
ggplotly(p_topic)

# ------------------------------------------------------------
# Step 5: Sentiment Analysis (NRC)
# ------------------------------------------------------------
# Note: You may need to run textdata::lexicon_nrc() once to download
nrc <- get_sentiments("nrc")

tidy_sent <- tidy_speeches %>%
  inner_join(nrc, by = "word")

sentiment_summary <- tidy_sent %>%
  count(president, sentiment, sort = TRUE)

# ------------------------------------------------------------
# Step 6: Hierarchical Clustering (The Fix is here)
# ------------------------------------------------------------
dtm_tfidf <- DocumentTermMatrix(Corpus(VectorSource(speeches$text)),
                                control = list(weighting = weightTfIdf))

dtm_sparse <- removeSparseTerms(dtm_tfidf, 0.99)
dist_matrix <- dist(as.matrix(dtm_sparse), method = "euclidean")
fit <- hclust(dist_matrix, method = "ward.D2")

# DYNAMIC CLUSTER CALCULATION
# This prevents the "k must be between 2 and 3" error
num_documents <- nrow(as.matrix(dtm_sparse))

# Set k to 4, but if you have fewer than 4 documents, set it to num_documents - 1
k_to_draw <- ifelse(num_documents > 4, 4, num_documents - 1)

# Ensure the plot is created
plot(fit, main = "Cluster Dendrogram of Presidents", labels = speeches$president)

# Only draw rectangles if we have enough groups to actually split
if(num_documents >= 2 && k_to_draw >= 2) {
  rect.hclust(fit, k = k_to_draw, border = "red") 
}

# ------------------------------------------------------------
# Step 7: Sentiment Volatility
# ------------------------------------------------------------
volatility <- tidy_speeches %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(president) %>%
  summarise(
    sentiment_mean = mean(value),
    sentiment_volatility = sd(value),
    word_count = n()
  ) %>%
  filter(word_count > 5) 

p_volatility <- ggplot(volatility, aes(x = sentiment_mean, y = sentiment_volatility, label = president)) +
  geom_point(color = "purple", size = 3, alpha = 0.7) +
  geom_text(vjust = 1.5, check_overlap = TRUE, size = 3) +
  labs(title = "Emotional Stability of Presidents",
       x = "Average Sentiment Score",
       y = "Volatility (Emotional Swings)") +
  theme_minimal()

ggplotly(p_volatility)

# ------------------------------------------------------------
# Step 8: Export results
# ------------------------------------------------------------
write.csv(top_terms, "LDA_TopTerms.csv", row.names = FALSE)
write.csv(sentiment_summary, "Sentiment_NRC.csv", row.names = FALSE)