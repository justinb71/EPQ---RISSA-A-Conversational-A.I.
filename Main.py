#Importing Libraries 
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import wolframalpha
import customtkinter
import tkinter as tk
from tkinter import *
from tkinter import ttk
from ttkthemes import ThemedStyle
import sv_ttk
from tkinter.scrolledtext import *
from PIL import Image, ImageTk

# Variables & Constants

currentUserName = "None"
botsName = "RISSA"
response_num = 0
changeName = False

# Wolfram Function
def wolframCalculator(question):
    app_id = "XG763R-2VQ83KP349"
    client = wolframalpha.Client(app_id)
    result = client.query(question)
    answer = next(result.results).text
    return answer

# Tokenizer
tokenizer2 = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
nlp = pipeline('ner', model=model, tokenizer=tokenizer2, aggregation_strategy="simple")

#Getting users name from text function
def getName(message):
    classifications = nlp(message)

    for i in range(len(classifications)):
        if classifications[i-1]["entity_group"] =="PER":
            name = classifications[i-1]["word"]

    return name




class IntentClassifier:
    def __init__(self,classes,model,tokenizer,label_encoder):
        self.classes = classes
        self.classifier = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def get_intent(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence)
        return self.label_encoder.inverse_transform(np.argmax(self.pred,1))[0]

intentsModel = load_model('models/intents.h5')

with open('utils/classes.pkl','rb') as file:
  classes = pickle.load(file)

with open('utils/tokenizer.pkl','rb') as file:
  tokenizer = pickle.load(file)

with open('utils/label_encoder.pkl','rb') as file:
  label_encoder = pickle.load(file)

nlu = IntentClassifier(classes,intentsModel,tokenizer,label_encoder)
intent = nlu.get_intent("Test")

def process():
    global changeName
    global name
    global currentUserName

    if changeName:
        remember = userInput.get()

        UserinputMessage()

        if "yes" in remember.lower() or "yeah" in remember.lower():
                currentUserName = str(name)
                addMessage("Ok, I will now call you " + currentUserName,False)
        

       
    else:
        message = userInput.get()
        UserinputMessage()

        intent = nlu.get_intent(message)

        if intent == ("what_is_your_name"):
            output = "My name is " + botsName

            user_input = conversationTokenizer.encode("What is your name" + conversationTokenizer.eos_token,return_tensors="pt")
            chatbot_input = torch.cat([chat_history, user_input], dim=-1) if response_num > 0 else user_input
            
            chat_history = conversationModel.generate(
                chatbot_input,
                max_length=1000,
                do_sample=True,
                top_k=70,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=conversationTokenizer.eos_token_id,
            )

        elif message == botsName:
            output = "That's my name"
        elif intent == "calculator":

            #Feeds the users input into the wolframAlpha API
            output = wolframCalculator(message)
        elif intent == "change_user_name":
            changeName = True
            name = getName(message)
            addMessage("Do you want me to remember you as " + str(name) + " ",False)
            
            

            

            
        else:
            
            
            user_input = conversationTokenizer.encode(message + conversationTokenizer.eos_token,return_tensors="pt")
            chatbot_input = torch.cat([chat_history, user_input], dim=-1) if response_num > 0 else user_input
            
            chat_history = conversationModel.generate(
                chatbot_input,
                max_length=1000,
                do_sample=True,
                top_k=70,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=conversationTokenizer.eos_token_id,
            )

            
            
            # print the output
            output = conversationTokenizer.decode(chat_history[:, chatbot_input.shape[-1]:][0], skip_special_tokens=True)
            

            

        addMessage(output, False)
        app.update()



model_name = "microsoft/DialoGPT-large"

conversationTokenizer = AutoTokenizer.from_pretrained(model_name)
conversationModel = AutoModelForCausalLM.from_pretrained(model_name)
conversationTokenizer.padding_side = "left"




customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()
app.geometry("420x600")
app.title("RISSA")
app.iconbitmap(default='favicon.ico')
app.resizable(False, False)



app.config(bg="#1c1c1c")


title = customtkinter.CTkLabel(app,text="RISSA", font=("Helveticca",26),bg_color="#1c1c1c")
title.pack(pady=10,anchor = customtkinter.CENTER)

frame = customtkinter.CTkFrame(master=app,bg_color="#1c1c1c")
frame.pack(padx=10, fill="both", expand = True)

def OnMouseWheel(self,event):
    self.scrollbar.yview("scroll",event.delta,"units")
    return "break" 

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self,background="#1c1c1c", highlightthickness=0)
        

     
        sv_ttk.set_theme("dark")


        self.scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = customtkinter.CTkFrame(self.canvas)

        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.bind(
            "<MouseWheel>",
            lambda event: self.yview_scroll(-1, "units")
        )

        self.scrollable_frame.update()
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor=tk.CENTER,width=375,height = 460)
        

        self.bind("<MouseWheel>", self.canvas.yview)
        

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right",fill = "y")

messageFrame = ScrollableFrame(frame)
messageFrame.pack(fill="both", expand=True)


userInput = customtkinter.CTkEntry(app,placeholder_text="Type To Speak")
userInput.pack(fill="x",padx=10,pady=5)

frameFillTotal = 0
def addMessage(message, userInput):
    global frameFillTotal

    if userInput == True:

        newMessage = customtkinter.CTkLabel(master=messageFrame.scrollable_frame,text=message, corner_radius= 20, fg_color="#4d4dcc",font=("Helveticca", 13), wraplength=200)
        newMessage.pack(padx = 20,pady=10, anchor = tk.E)
        

        
    else:
        newMessage = customtkinter.CTkLabel(master=messageFrame.scrollable_frame,text=message,corner_radius= 20, fg_color="#aa4241",font=("Helveticca", 13), wraplength=200)
        newMessage.pack(padx=20 ,pady=10,anchor = tk.W)

    messageFrame.scrollable_frame.update()
    newMessage.update()
    frameFillTotal = frameFillTotal + newMessage.winfo_reqheight()
    if frameFillTotal >=450:
        messageFrame.canvas.create_window((0, 0), window=messageFrame.scrollable_frame, anchor=tk.CENTER,width=375,height = 460 + newMessage.winfo_reqheight())
    
        messageFrame.canvas.yview_moveto(1)

    


def UserinputMessage():
    addMessage(userInput.get(),True)
    userInput.delete(0, customtkinter.END)

button = customtkinter.CTkButton(master=app, text="Send", fg_color="#444444",command = process, font=("Helveticca", 15))
button.pack(anchor=customtkinter.CENTER, pady=10)

addMessage("Hey",False)
app.mainloop()



