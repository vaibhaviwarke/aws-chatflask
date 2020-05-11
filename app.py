from flask import Flask, render_template, request, Response
import nltk
import numpy as np
import random
import string # to process standard python strings
from werkzeug.utils import secure_filename           #for image input
import os
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
import time
import sys
import numpy as np

app = Flask(__name__)

f=open('chatbot.txt','r',errors = 'ignore')
raw=f.read()

raw=raw.lower()# converts to lowercase
nltk.download('punkt') # first-time use for english
nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
	

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
			
		
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_bot_response(user_response):
    robo_response=''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    print(sent_tokens)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    print(tfidf[-1])   								 #userinput tfidf
    print("*********************")
    print(tfidf)       								 #tfidf of whole database
    print("Cosine ")
    print(vals)       							     #cosine similarity value of every document
    idx=vals.argsort()[0][-2]						 #second largest cosine similarity value's index is stored in idx
    print(idx)										 #idx
    flat = vals.flatten()							 #converts 2d array into 1d array
    print("flat")
    print(flat)                                      #1d array
    flat.sort()
    req_tfidf = flat[-2]							# second largest tfidf
    print("REQ tifid")
    print(req_tfidf)
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

def text_response(user_response):		
 flag=True
 print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
 while(flag==True):
    print(user_response)
    #print(type(user_response))
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
                return greeting(user_response)
            else:
                word_tokens=[]
                sent_tokens.append(user_response)
                word_tokens+=nltk.word_tokenize(user_response)
                final_words=list(set(word_tokens))
                print(word_tokens)
                print("******************************")
                print(final_words)
                print("ROBO: ",end="")
                global output
                output=str(get_bot_response(user_response))
                print(output)
                print(type(output))
                sent_tokens.remove(user_response)
                return output
				
    else:
        flag=False
        print("ROBO: Bye! take care..")


def fetchnumber(output):
  test_string =output
  print("The original string : " + test_string) 
  #res = [int(i) for i in test_string.split() if i.isdigit()] 
  for i in test_string.split():
        if i.isdigit():
            res=int(i)
  print("The number  is : "+str(res)) 
  return res
  
'''from twilio.rest import Client as Call

def Calling(number):
  print("Inside Call Function")
  From_Number = "+12028901465"
  numb="+91"+str(number)
  no=int(numb)
  To_Number = no  #provide an emergency number in any variable to from_number e.g firebriged calling number
  Src_Path = "http://static.fullstackpython.com/phone-calls-python.xml"  #add here the

  client = Call("AC0ec3e7a01f011fc686aaf5dcd91c5881" , "2cebe4f3021a076998417b7625992b99")
  print('call initialted')
  client.calls.create(to = To_Number, from_=From_Number, url = Src_Path, method = 'GET')
  print('Call has been triggered successfully')'''

      
def imageprocess(imagename):
    new_model = tf.keras.models.load_model('saved_model/my_model1')

    #Check its architecture
    #new_model.summary()
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img(imagename, target_size = (64, 64))   #file to be predicted imagename
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = new_model.predict(test_image)
    print(result)

    if result[0][0] == 0:
       prediction = 'Fire brigade number'
    else:
       prediction = 'Smoke'
	
    return prediction
    
    
#define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/capture", methods=['POST'])
def capture():
    return Response(open('./static/getImage.html').read(), mimetype="text/html")


# save the image as a picture
@app.route('/image', methods=['POST'])
def image():

    i = request.files['image']  # get the image
    f = ('%s.jpeg' % time.strftime("%Y%m%d-%H%M%S"))
    i.save('%s/%s' % (PATH_TO_TEST_IMAGES_DIR, f))

    return Response("%s saved" % f)
	
@app.route("/get")
#function for the bot response
def text():
     input=request.args.get('msg')
     #print(input)
     return text_response(input)
    
 
@app.route('/voice')
def get_voice():
     input=request.args.get('voice')
     #print(input)
     return text_response(input)


@app.route('/call', methods=['POST'])
def calls():
    print(request.method)
    #print(output)
    if request.method=="POST":
        #number=fetchnumber(output)
        #Calling(number)
        return "Call Successful"
        #return render_template("mo.html",value=number)


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   #global secure_filename
   if request.method == 'POST':
      f = request.files['file']
      secure_filename1 = secure_filename(f.filename)
      img_path=os.path.join(app.root_path,secure_filename1)
      f.save(img_path)
      print("File name printed")
      print(f.filename)
      print(secure_filename1)
      pred = imageprocess(secure_filename1)
      return render_template("imageoutput.html",value=pred)
      
     
if __name__ == "__main__":
    app.debug=True
    app.run()
