import nltk
nltk.download('punkt')
nltk.download('stopwords')
 # this gives us a list of sentences
from gensim.utils import tokenize

from nltk.corpus import stopwords
import pickle

import textstat


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from pprint import pprint


stop_words = set(stopwords.words('english'))

VECTORIZER_FILE = "static/vectorizer.pkl"

def preProcess(text, sentSep=" ", sentMap=False):
    sent_text = nltk.sent_tokenize(text)
    sent_text_lower = [t.lower() for t in sent_text]
    sentenceMapping = {}
    # tokens = tokenize(text)
    tok_sent_text = []
    for i in range(len(sent_text_lower)):
        tokens = list(tokenize(sent_text_lower[i]))
        filtered_sentence_words = [w for w in tokens if not w in stop_words and len(w)>2]
        sentenceText = " ".join(filtered_sentence_words)
        tok_sent_text.append(sentenceText)
        if sentMap:
            sentenceMapping[i] = sent_text[i]
    finalText = sentSep.join(tok_sent_text)
    return finalText, sentenceMapping

def getSentenceScores(sentences, sentenceMapping):
    vectorizer = None
    with open(VECTORIZER_FILE, "rb") as f:
        vectorizer = pickle.load(f)
    text = " ".join(sentences).lower()
    tmp = vectorizer.transform([text])
    dense = tmp.todense()
    denselist = dense.tolist()
    feature_names = vectorizer.get_feature_names()
    df = pd.DataFrame(denselist, columns=feature_names)
    scores = []
    for index, i in enumerate(sentences):
        score = 0
        for j in i.lower().split(" "):
            try:
                score += df[j][0]
            except KeyError:
                score += 0
        scores.append((sentenceMapping[index], score, index))
    return scores



def getSummary(text,percent, sentenceMapping):
    # print(sentenceMapping)
    length = percent/100
    summary_len = int(len(text)*length)
    summary = []
    summary_scores = text[:summary_len]
    summary_scores = sorted(summary_scores, key=lambda x: x[2], reverse=False)
    # pprint(summary_scores)
    # for sent in summary_scores:
    for i in range(len(summary_scores)):
        sentenceIndex = summary_scores[i][2]
        sentence = sentenceMapping[sentenceIndex]
        # sentence = summary_scores[i][0]
        # summary += (". "+(sent[0]))
        summary.append(sentence)
    summaryLen = len(summary)
    summary = ". ".join(summary)
    return summary, summaryLen


def summarizer(text,sumLen):
  tmp, sentMap = preProcess(text, sentSep=":::", sentMap=True)
  originalLen = len(sentMap)
  scores = getSentenceScores(tmp.split(':::'), sentMap)
  ranked_sentence = sorted(scores, key=lambda x: x[1], reverse=True)
  # pprint(ranked_sentence)
  summary, summaryLen = getSummary(ranked_sentence, sumLen, sentMap)
  return summary, originalLen, summaryLen


def readScore(text):
  score = textstat.flesch_reading_ease(text)
  return score


# txt = None
# with open("static/tstPaper.txt", encoding="utf8") as f:
#     txt = f.read()

# ans = readScore(txt)
# print(ans)

# summary = summarizer(txt, 1)
# print(summary)
