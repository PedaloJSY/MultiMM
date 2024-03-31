# MultiMM
All data in the master branch！

We introduce a Multimodal Multicultural Metaphor dataset (MultiMM) to facilitate an extensive cross-cultural study of metaphor in Chinese and English. It contains 8,461 text-image pairs of advertisements with manual annotations of the occurrence of metaphors, target/source vocabulary, and sentiments metaphors convey.

# Data
In the data files EN.csv and ZH.csv, the "Pic_id" column identifies the unique picture identifier; the "Text" column contains the corresponding textual content; the "MetaphorOccurrence" column indicates whether the text includes a metaphor, with 0 representing literal meaning in the image and text, and 1 indicating the presence of metaphorical meaning; the "Source" column points out the source domain in the metaphorical expression, while the "Target" column represents the target domain; the "SentimentCategory" column reflects the sentiment tendency, with -1 representing negative sentiment, 0 for neutral sentiment, and 1 for positive sentiment.

# Code
The "data_utils.py" file contains functions for data reading, processing, and the implementation of related evaluation metrics; the "model.py" file constitutes the main network structure of the model; and the "main.py" file primarily includes functions for training the model and the relevant parameter settings.
