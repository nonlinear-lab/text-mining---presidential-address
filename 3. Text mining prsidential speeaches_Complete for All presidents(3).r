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
setwd("C:/R/speech")

files <- list.files(pattern = "*.txt")
if (length(files) == 0) stop("No .txt files found in the directory!")

speeches <- map_df(files, function(f) {
  text <- readLines(f, encoding = "UTF-8", warn = FALSE)
  tibble(
    president = tools::file_path_sans_ext(f),
    text      = paste(text, collapse = " ")
  )
})

cat("Loaded", nrow(speeches), "speeches.\n")

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
  count(president, word, sort = TRUE)

# FIX: Open a dedicated graphics window so wordcloud is never overwritten
dev.new(width = 10, height = 8)
set.seed(123)
wordcloud(
  words        = word_freq$word,
  freq         = word_freq$n,
  max.words    = 100,
  random.order = FALSE,
  colors       = brewer.pal(8, "Dark2"),
  scale        = c(4.5, 0.5)
)
title("Word Cloud: Presidential Inaugural Speeches")

write.csv(word_freq, "freq.csv", row.names = FALSE)

# ------------------------------------------------------------
# Step 4: Topic Modeling (LDA)
# ------------------------------------------------------------
dtm <- tidy_speeches %>%
  count(president, word) %>%
  cast_dtm(document = president, term = word, value = n)

# Dynamically set k so it never exceeds number of documents
n_docs <- nrow(as.matrix(dtm))
lda_k  <- min(5, n_docs)

set.seed(123)
lda <- LDA(dtm, k = lda_k, method = "VEM")

topics_tidy <- tidy(lda, matrix = "beta")
gamma_tidy  <- tidy(lda, matrix = "gamma")

top_terms <- topics_tidy %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup()

p_topic <- ggplot(top_terms,
                  aes(x = reorder_within(term, beta, topic),
                      y = beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free_y") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Top Terms per Topic (LDA)", x = "Term", y = "Probability") +
  theme_minimal()

print(ggplotly(p_topic))  # FIX: explicit print()

# ------------------------------------------------------------
# Step 5: Sentiment Analysis (NRC)
# ------------------------------------------------------------
nrc <- get_sentiments("nrc")

tidy_sent <- tidy_speeches %>%
  inner_join(nrc, by = "word")

sentiment_summary <- tidy_sent %>%
  count(president, sentiment, sort = TRUE)

p_sent <- ggplot(sentiment_summary,
                 aes(x = reorder_within(sentiment, n, president),
                     y = n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ president, scales = "free_y") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Sentiment Distribution per President (NRC)",
       x = "Sentiment", y = "Word Count") +
  theme_minimal()

print(ggplotly(p_sent))  # FIX: explicit print()

# ------------------------------------------------------------
# Step 6: Topic-Sentiment Correlation
# ------------------------------------------------------------
topic_sent <- top_terms %>%
  inner_join(nrc, by = c("term" = "word")) %>%
  count(topic, sentiment, sort = TRUE)

p_topic_sent <- ggplot(topic_sent,
                       aes(x = sentiment, y = n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free_y") +
  coord_flip() +
  labs(title = "Sentiment Distribution per Topic",
       x = "Sentiment", y = "Frequency") +
  theme_minimal()

print(ggplotly(p_topic_sent))  # FIX: explicit print()

gamma_sent <- gamma_tidy %>%
  rename(president = document) %>%
  left_join(tidy_sent %>% count(president, sentiment), by = "president") %>%
  group_by(topic, sentiment) %>%
  summarise(avg_gamma    = mean(gamma, na.rm = TRUE),
            total_sent   = sum(n, na.rm = TRUE),
            .groups = "drop") %>%
  mutate(topic_sent_score = avg_gamma * total_sent)

p_heat <- ggplot(gamma_sent,
                 aes(x = factor(topic), y = sentiment, fill = topic_sent_score)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightyellow", high = "darkred") +
  labs(title = "Topic-Sentiment Correlation Heatmap",
       x = "Topic", y = "Sentiment", fill = "Score") +
  theme_minimal()

print(ggplotly(p_heat))  # FIX: explicit print()

# ------------------------------------------------------------
# Step 7: Bigram Analysis
# ------------------------------------------------------------
speeches_bigrams <- speeches %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2)

bigrams_separated <- speeches_bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")

bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !str_detect(word1, "\\d"),
         !str_detect(word2, "\\d"))

bigram_counts <- bigrams_filtered %>%
  count(word1, word2, sort = TRUE)

bigrams_united <- bigrams_filtered %>%
  unite(bigram, word1, word2, sep = " ")

p_bigram <- bigrams_united %>%
  count(bigram, sort = TRUE) %>%
  slice_max(n, n = 20) %>%
  mutate(bigram = reorder(bigram, n)) %>%
  ggplot(aes(n, bigram, fill = n)) +
  geom_col(show.legend = FALSE) +
  labs(title = "Top 20 Bigrams (Two-Word Phrases)", x = "Frequency", y = NULL) +
  theme_minimal()

print(ggplotly(p_bigram))  # FIX: explicit print()

# ------------------------------------------------------------
# Step 8: Network Graph of Co-occurring Words
# ------------------------------------------------------------
bigram_graph <- bigram_counts %>%
  filter(n > 5) %>%
  graph_from_data_frame()

set.seed(2023)
ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                 arrow = grid::arrow(type = "closed", length = unit(.1, "inches")),
                 end_cap = circle(.07, "inches")) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void() +
  labs(title = "Network of Word Co-occurrences")

# ------------------------------------------------------------
# Step 9: TF-IDF (Distinctive Words per President)
# ------------------------------------------------------------
president_tfidf <- tidy_speeches %>%
  count(president, word, sort = TRUE) %>%
  bind_tf_idf(word, president, n)

p_tfidf <- president_tfidf %>%
  group_by(president) %>%
  slice_max(tf_idf, n = 10) %>%
  ungroup() %>%
  filter(president %in% unique(president)[1:4]) %>%
  ggplot(aes(tf_idf, reorder_within(word, tf_idf, president), fill = president)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ president, scales = "free") +
  scale_y_reordered() +
  labs(title = "Most Distinctive Words (TF-IDF)", x = "TF-IDF Score", y = NULL) +
  theme_minimal()

print(ggplotly(p_tfidf))  # FIX: explicit print()

# ------------------------------------------------------------
# Step 10: Lexical Diversity & Complexity
# ------------------------------------------------------------
sentence_complexity <- speeches %>%
  mutate(sentences           = str_count(text, "[.?!]"),
         word_count          = str_count(text, "\\w+"),
         avg_sentence_length = word_count / sentences)

lexical_diversity <- tidy_speeches %>%
  group_by(president) %>%
  summarise(unique_words      = n_distinct(word),
            total_words       = n(),
            lexical_diversity = unique_words / total_words) %>%
  inner_join(sentence_complexity, by = "president")

p_complexity <- ggplot(lexical_diversity,
                       aes(x = avg_sentence_length, y = lexical_diversity,
                           label = president)) +
  geom_point(color = "darkblue", size = 3, alpha = 0.7) +
  geom_text(vjust = 1.5, check_overlap = TRUE, size = 3) +
  labs(title = "Speech Complexity: Diversity vs. Sentence Length",
       x = "Avg Words per Sentence",
       y = "Lexical Diversity (Unique Word Ratio)") +
  theme_minimal()

print(ggplotly(p_complexity))  # FIX: explicit print()

# ------------------------------------------------------------
# Step 11: Emotional Arch — Sentiment Narrative Arc
# (YOUR ORIGINAL APPROACH — preserved exactly, with print() fix)
# ------------------------------------------------------------
narrative_arc <- tidy_speeches %>%
  inner_join(get_sentiments("bing"), by = "word") %>%
  count(president, index = 1:n() %/% 40, sentiment) %>%  # groups every 40 words
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment_score = positive - negative) %>%
  group_by(president) %>%
  mutate(pct_progress = row_number() / n() * 100) %>%  # normalize to 0-100%
  ungroup()

# Show all presidents (not just first 5)
target_presidents <- unique(speeches$president)

p_flow <- narrative_arc %>%
  filter(president %in% target_presidents) %>%
  ggplot(aes(x = pct_progress, y = sentiment_score, color = president)) +
  geom_smooth(method = "loess", se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  labs(title    = "Emotional Arc of Speeches",
       subtitle = "Smoothed Sentiment from Start (0%) to End (100%)",
       x        = "Progress through Speech (%)",
       y        = "Sentiment Score (Positive - Negative)") +
  theme_minimal()

print(ggplotly(p_flow))  # FIX: explicit print()

# ------------------------------------------------------------
# Step 12: Hierarchical Clustering (Dendrogram)
# ------------------------------------------------------------
dtm_tfidf  <- DocumentTermMatrix(
  Corpus(VectorSource(speeches$text)),
  control = list(weighting = weightTfIdf)
)
dtm_sparse  <- removeSparseTerms(dtm_tfidf, 0.95)
dist_matrix <- dist(as.matrix(dtm_sparse), method = "euclidean")
fit         <- hclust(dist_matrix, method = "ward.D2")

num_documents <- nrow(as.matrix(dtm_sparse))
k_to_draw     <- ifelse(num_documents > 4, 4, max(2, num_documents - 1))

# FIX: Open a new device so dendrogram is not overwritten by wordcloud
dev.new(width = 12, height = 7)
plot(fit,
     main   = "Cluster Dendrogram of Presidents' Speeches",
     xlab   = "President",
     sub    = "",
     cex    = 0.8)

if (num_documents >= 2 && k_to_draw >= 2) {
  rect.hclust(fit, k = k_to_draw, border = "red")
}

# ------------------------------------------------------------
# Step 13: Phrase Net ("of", "and", "to" patterns)
# ------------------------------------------------------------
phrase_net_data <- speeches %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(word1 %in% c("of", "and", "to") | word2 %in% c("of", "and", "to")) %>%
  count(word1, word2, sort = TRUE) %>%
  filter(n > 10) %>%
  graph_from_data_frame()

set.seed(2024)
ggraph(phrase_net_data, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n, edge_width = n), edge_colour = "darkred",
                 arrow = grid::arrow(type = "closed", length = unit(2, "mm")),
                 end_cap = circle(3, "mm")) +
  geom_node_point(size = 4, color = "black") +
  geom_node_text(aes(label = name), vjust = 1.5, repel = TRUE) +
  theme_void() +
  labs(title    = "Phrase Net: How Words Are Linked",
       subtitle = "Visualizing connectors: 'of', 'and', 'to'")

# ------------------------------------------------------------
# Step 14: Sentiment Volatility (Emotional Rollercoaster Index)
# ------------------------------------------------------------
volatility <- tidy_speeches %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(president) %>%
  summarise(
    sentiment_mean       = mean(value),
    sentiment_volatility = sd(value),
    word_count           = n()
  ) %>%
  filter(word_count > 50)

p_volatility <- ggplot(volatility,
                       aes(x = sentiment_mean, y = sentiment_volatility,
                           label = president)) +
  geom_point(color = "purple", size = 3, alpha = 0.7) +
  geom_text(vjust = 1.5, check_overlap = TRUE, size = 3) +
  labs(title = "Emotional Stability of Presidents",
       x     = "Average Sentiment (Positive/Negative)",
       y     = "Volatility (Emotional Swings)") +
  theme_minimal()

print(ggplotly(p_volatility))  # FIX: explicit print()

# ------------------------------------------------------------
# Step 15: Export Results
# ------------------------------------------------------------
write.csv(top_terms,    "LDA_TopTerms.csv",              row.names = FALSE)
write.csv(sentiment_summary, "Sentiment_NRC.csv",        row.names = FALSE)
write.csv(topic_sent,   "Topic_Sentiment.csv",           row.names = FALSE)
write.csv(gamma_sent,   "Topic_Sentiment_Correlation.csv", row.names = FALSE)
write.csv(narrative_arc, "Emotional_Arc.csv",            row.names = FALSE)

cat("\nDone! All figures displayed and CSV files saved.\n")
