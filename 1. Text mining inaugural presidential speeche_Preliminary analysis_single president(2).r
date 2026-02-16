# ============================================================
# TEXT MINING: Inaugural Presidential Speech (Full Fixed Code)
# ============================================================

# Load required libraries
library(tidyverse)
library(tidytext)
library(tm)
library(wordcloud)
library(SnowballC)
library(sentimentr)
library(ggplot2)
library(textdata)
library(dplyr)
library(stringr)
library(tidyr)
library(topicmodels)
library(widyr)
library(ggraph)
library(igraph)
library(tidygraph)

# ------------------------------------------------------------
# Step 1: Set working directory and load text
# ------------------------------------------------------------
setwd("C:/R/speech") 

# Load the speech text file
speech <- readLines("DTrump.txt", encoding = "UTF-8")
speech_text <- paste(speech, collapse = " ")
speech_df <- tibble(line = 1, text = speech_text)

# ------------------------------------------------------------
# Step 2 & 3: Tokenize and Clean
# ------------------------------------------------------------
data("stop_words")

clean_tokens <- speech_df %>%
  unnest_tokens(word, text) %>%
  mutate(word = str_replace_all(word, "[^a-zA-Z]", "")) %>%
  filter(word != "", nchar(word) > 2) %>%
  anti_join(stop_words, by = "word")

# ------------------------------------------------------------
# Step 4: Word frequency count
# ------------------------------------------------------------
word_freq <- clean_tokens %>%
  count(word, sort = TRUE)

# ------------------------------------------------------------
# Step 5: Word Cloud (Saves and Displays)
# ------------------------------------------------------------
# Save to file
png("1_wordcloud.png", width=800, height=800)
set.seed(1234)
wordcloud(words = word_freq$word, freq = word_freq$n, max.words = 100, colors = brewer.pal(8, "Dark2"))
dev.off() 

# Show in RStudio
set.seed(1234)
wordcloud(words = word_freq$word, freq = word_freq$n, max.words = 100, colors = brewer.pal(8, "Dark2"))

# ------------------------------------------------------------
# Step 6: Bar Chart of Top 15 Words
# ------------------------------------------------------------
p_freq <- word_freq %>%
  top_n(15, n) %>%
  ggplot(aes(x = reorder(word, n), y = n, fill = n)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  labs(title = "Most Frequent Words", x = "Words", y = "Frequency") +
  theme_minimal()

print(p_freq) 
ggsave("2_word_frequency.png", p_freq, width = 8, height = 6)

# ------------------------------------------------------------
# Step 7: Sentiment Analysis (Restored and Completed)
# ------------------------------------------------------------
nrc_lexicon <- get_sentiments("nrc")

speech_sentiment <- clean_tokens %>%
  inner_join(nrc_lexicon, by = "word", relationship = "many-to-many") %>%
  count(sentiment, sort = TRUE)

p_sent <- ggplot(speech_sentiment, aes(x = reorder(sentiment, n), y = n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  labs(title = "Sentiment Analysis (NRC Lexicon)", x = "Sentiment Category", y = "Word Count") +
  theme_minimal()

print(p_sent) 
ggsave("3_sentiment_analysis.png", p_sent, width = 8, height = 6)

# ------------------------------------------------------------
# Step 8: Bigram Analysis (Restored and Completed)
# ------------------------------------------------------------
bigrams_separated <- speech_df %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word)

bigram_counts <- bigrams_separated %>%
  unite(bigram, word1, word2, sep = " ") %>%
  count(bigram, sort = TRUE)

p_bigram <- bigram_counts %>%
  top_n(10, n) %>%
  ggplot(aes(x = reorder(bigram, n), y = n, fill = n)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  labs(title = "Top Word Pairs (Bigrams)", x = "Word Pair", y = "Frequency") +
  theme_minimal()

print(p_bigram)
ggsave("4_bigram_analysis.png", p_bigram, width = 8, height = 6)

# ------------------------------------------------------------
# Step 9: Word Network (Restored and Completed)
# ------------------------------------------------------------
bigram_graph <- bigrams_separated %>%
  count(word1, word2, sort = TRUE) %>%
  filter(n > 1) %>%
  as_tbl_graph()

p_net <- ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  labs(title = "Word Network Analysis") +
  theme_void()

print(p_net)
ggsave("5_word_network.png", p_net, width = 10, height = 8)