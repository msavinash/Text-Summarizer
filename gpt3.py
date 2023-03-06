import os
import openai

import nltk
nltk.download('punkt')
nltk.download('stopwords')
 # this gives us a list of sentences
from gensim.utils import tokenize

from nltk.corpus import stopwords
import pickle
import textstat
import pandas as pd
import re
from rouge import Rouge

### Add your OpenAI API Key here
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

stop_words = set(stopwords.words('english'))

VECTORIZER_FILE = "static/vectorizer.pkl"

def preProcess(text, sentSep=" ", sentMap=False):
    text = text.lower()
    sent_text = nltk.sent_tokenize(text)
    sentenceMapping = {}
    # tokens = tokenize(text)
    tok_sent_text = []
    for i in range(len(sent_text)):
        tokens = list(tokenize(sent_text[i]))
        filtered_sentence_words = [w for w in tokens if not w in stop_words and len(w)>2]
        sentenceText = " ".join(filtered_sentence_words)
        tok_sent_text.append(sentenceText)
        if sentMap:
            sentenceMapping[i] = sent_text[i]
    finalText = sentSep.join(tok_sent_text)
    return finalText, sentenceMapping

# gpt-3 summarization
def summary_gpt3(input_text, percent, sentSep=" ", sentMap=False):
    percentage = percent * 2
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Given a transcript, get summary:\n\nTranscript: {input_text}\n\nSummary:",
        temperature=0,
        max_tokens=int(percentage), #change summary length
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    summary = response["choices"][0]["text"]
    final_summary = summary.replace("\n","").strip()
    
    count = re.split(r'[.!?]+', final_summary)
    summary_len = len(count)

    return final_summary, summary_len

def gpt3_readScore(text):
  score = textstat.flesch_reading_ease(text)
  return score

# F1, recall, and precision
def metrics(orginal_text, final_text):
    ROUGE = Rouge()
    rogue_score = ROUGE.get_scores(orginal_text, final_text)

    return rogue_score

def gpt3_summarizer(text, sumLen):
  clean_text, sentMap = preProcess(text, sentSep=". ", sentMap=True)
  originalLen = len(sentMap)
  summary, summary_len = summary_gpt3(clean_text, sumLen)
  return summary, summary_len, originalLen