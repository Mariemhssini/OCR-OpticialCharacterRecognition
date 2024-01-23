import pickle
import string  
import re
from flask import Flask, jsonify, request
import nltk
import warnings
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
st = ISRIStemmer()
nltk.download('stopwords')
stop=stopwords.words('arabic')

def clean(text):
  #remove all English chars 
  text = re.sub(r'\s*[A-Za-z]\s*', ' ' , text)
  #remove hashtags
  text = re.sub("#", " ", text)
  #remove all numbers 
  text = re.sub(r'\[0-9]*\]',' ',text)
  #remove duplicated chars
  text = re.sub(r'(.)\1+', r'\1', text)
  #remove :) or :(
  text = text.replace(':)', "")
  text = text.replace(':(', "")
  #remove multiple exclamation
  text = re.sub(r"(\!)\1+", ' ', text)
  #remove multiple question marks
  text = re.sub(r"(\?)\1+", ' ', text)
  #remove multistop
  text = re.sub(r"(\.)\1+", ' ', text)
  #remove additional spaces
  text = re.sub(r"[\s]+", " ", text)
  text = re.sub(r"[\n]+", " ", text)
  
  return text

def remStopWords(Text):
  return " ".join(word for word in Text.split() if word not in stop)

def stemWords(Text):
  return " ".join(st.stem(word) for word in Text.split())
# Load the SVM classifier
with open('Pickles\TextClassifier.pkl', 'rb') as pklClassifierFile:
    clf_svm = pickle.load(pklClassifierFile)

# Load the label encoder
with open('Pickles\LabelEncoder.pkl', 'rb') as pklEncodingFile:
    encoder = pickle.load(pklEncodingFile)

# Load the text vectorizer
with open('Pickles\TextVectorizer.pkl', 'rb') as pklVectorizerFile:
    tfidf_vect = pickle.load(pklVectorizerFile)

app = Flask(__name__)
warnings.filterwarnings("ignore")


def pipeline(Text):
    # Preprocessing steps
    Text = clean(Text)
    Text = "".join([char for char in Text if char not in string.ascii_letters]).strip()
    Text = remStopWords(str(Text))
    Text = stemWords(Text)
    # Vectorize the text
    Text_Vector = tfidf_vect.transform([Text])
    # Make predictions
    predictions = clf_svm.predict(Text_Vector)
    print(predictions)
    return encoder.inverse_transform(predictions)[0]


@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()
        print(data)
        text_input = data["text"]
        print(text_input)
        # Preprocessing and classification
        processed_text = pipeline(text_input)
        return jsonify({'result': processed_text})
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)