import textstat
import nltk
nltk.download('punkt')
nltk.download('stopwords')
 # this gives us a list of sentences
from gensim.utils import tokenize

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from simpletransformers.seq2seq import Seq2SeqModel,Seq2SeqArgs
import pandas as pd

from sklearn.model_selection import train_test_split

stop_words = set(stopwords.words('english'))

df_rp = pd.read_csv('data.csv')
df_rp = df_rp.iloc[: , 1:]


def readScore(text):
  score = textstat.flesch_reading_ease(text)
  return score

model_args = Seq2SeqArgs()
model_args.num_train_epochs = 3
model_args.no_save = True
model_args.per_device_train_batch_size=16,  # batch size per device during training
model_args.per_device_eval_batch_size=64,   # batch size for evaluation
model_args.logging_dir='./logs',            # directory for storing logs
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.max_length = 600
model_args.overwrite_output_dir = True

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    args=model_args,
    use_cuda=True,
)

train, test = train_test_split(df_rp, test_size=0.2, random_state=23, shuffle=True)

model.train_model(train, eval_data=test)

results = model.eval_model(test)

def bartSumm(text):
  strList = [text]
  new = model.predict(strList)
  return new

import pickle

with open("fineTunedBert.pkl", "wb") as f:
    pickle.dump(model, f)


'''
import matplotlib.pyplot as plt

epoch_num = [1,2,3,4]
eval_loss = [1.8538, 1.5481, 1.5256, 1.4887]
train_loss = [1.4479,0.7484,1.1166,1.9231]
plt.plot(epoch_num, eval_loss,label = "Eval loss")
plt.plot(epoch_num, train_loss,label = "Train Loss")
plt.title('Loss metric - 4 Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

epoch_num = [1,2,3]
eval_loss = [1.8876, 1.5338, 1.4814]
train_loss = [2.0538,1.3792,1.2718]
plt.plot(epoch_num, eval_loss,label = "Eval loss")
plt.plot(epoch_num, train_loss,label = "Train Loss")
plt.title('Loss metric - 3 Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

epoch_num = [1,2,3,4,5,6,7,8,9,10]
eval_loss = [1.9590632594548738,
1.5406644367254698,
1.4973345674001253,
1.4726831294023073,
1.491552125949126,
1.511670667391557,
1.5423877789424016,
1.5680834375894988,
1.5980445513358483,
1.6138401123193593]
train_loss = [1.874458909034729,
0.7891172766685486,
1.3528355360031128,
1.400651216506958,
1.1273165941238403,
1.2451975345611572,
0.7505534291267395,
1.0538867712020874,
0.5617988109588623,
0.450857013463974]
plt.plot(epoch_num, eval_loss,label = "Eval loss")
plt.plot(epoch_num, train_loss,label = "Train Loss")
plt.title('Loss metric - 10 Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''