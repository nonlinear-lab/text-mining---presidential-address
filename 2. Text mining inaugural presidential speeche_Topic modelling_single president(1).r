# ------------------------------------------------------------
# Topic Modeling: Inaugural Speech (Tidytext Version)
# ------------------------------------------------------------

# Load core libraries
library(tidyverse)
library(tidytext)
library(topicmodels)
library(quanteda)
library(LDAvis)
library(servr)

# 1. PREPARE DATA
# 'data_corpus_inaugural' is built into quanteda
speeches_raw <- tidy(data_corpus_inaugural)

# 2. CLEANING & TOKENIZATION
# Note: Use 'President' and 'Year' as the grouping columns
tidy_speeches <- speeches_raw %>%
  mutate(doc_id = paste(President, Year, sep = "_")) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words) %>%
  filter(!str_detect(word, "[0-9]")) %>% # Remove dates
  filter(nchar(word) > 3)                # Remove tiny words

# 3. CREATE DOCUMENT-TERM MATRIX (DTM)
speech_counts <- tidy_speeches %>%
  count(doc_id, word)

speech_dtm <- speech_counts %>%
  cast_dtm(doc_id, word, n)

# Safety check: Remove empty documents if any exist
speech_dtm <- speech_dtm[slam::row_sums(speech_dtm) > 0, ]

# 4. RUN TOPIC MODEL (LDA)
# k=4 topics usually splits 'War', 'Economy', 'Unity', and 'Founding Principles'
k <- 4
inaugural_lda <- LDA(speech_dtm, k = k, method = "Gibbs", 
                     control = list(seed = 1234, iter = 500))

# 5. PREPARE FOR LDAVIS (The "Bridge")
# Extracting necessary distributions from the model
phi <- as.matrix(posterior(inaugural_lda)$terms)
theta <- as.matrix(posterior(inaugural_lda)$topics)
vocab <- colnames(phi)
doc_length <- slam::row_sums(speech_dtm)
term_frequency <- slam::col_sums(speech_dtm)

# 6. GENERATE VISUALIZATION
json_lda <- createJSON(phi = phi, 
                       theta = theta, 
                       doc.length = doc_length, 
                       vocab = vocab, 
                       term.frequency = term_frequency)

# This will launch the interactive map in your browser
serVis(json_lda)