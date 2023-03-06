import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

 # this gives us a list of sentences
from gensim.utils import tokenize
import pickle

import textstat


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from pprint import pprint

VECTORIZER_FILE = "static/vectorizer.pkl"

def preProcess(text, sentSep=" ", sentMap=False):
    sent_text = spacy.sent_tokenize(text)
    sent_text_lower = [t.lower() for t in sent_text]
    sentenceMapping = {}
    # tokens = tokenize(text)
    tok_sent_text = []
    for i in range(len(sent_text_lower)):
        tokens = list(tokenize(sent_text_lower[i]))
        filtered_sentence_words = [w for w in tokens if not w in STOP_WORDS and len(w)>2]
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
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

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

text = '''Analytical marketing: How Turkcell reduced their sales cycle from weeks to days
There are an enormous number of solutions telecommunications companies can provide, but most customers are only interested in a select few. Analytical marketing is the practice of using customer data to identify which consumers will most likely benefit from which products. This involves using consumer information to create microsegments, or personas that describe smaller sets of customers rather than large, big-picture details. It also includes detecting specific behaviors that prompt a new product offer, such as suggesting an international data plan to a customer who frequently sends texts to friends in other countries.

The first step in leveraging analytical marketing is to gather data. Companies need to strike a balance between obtaining high-quality, comprehensive information and maintaining trust with the consumer.

With the increasing capabilities for data mining, consumers have become protective of their personal information. Shady practices like obtaining and using data without notifying the customer, or selling and buying information from third-party sources have left consumers scarred and wary.

Ironically, consumers also become frustrated when they perceive companies to not have a good understanding of their needs or offer random products rather than relevant ones. This complex dynamic means telecom businesses must gather high-quality information, but use it in a responsible, transparent way.

A 2011 marketing campaign completed by Turkcell, an Istanbul-based cellphone provider, demonstrated the successful use of consumer data. The company began reviewing customer data in real time, allowing them to better identify individuals’ needs at that time. As a result, they were able to shorten their marketing cycle from several weeks to several days and increase revenue by $15 million that year, according to a Strategy& report from PwC.'''


summary = summarizer(text, 1)
print(summary)
