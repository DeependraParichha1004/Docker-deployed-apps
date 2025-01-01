#!/usr/bin/env python
# coding: utf-8

# In[20]:


import streamlit as st
import torchaudio
import torch
import soundfile as sf
import io

device = ("cuda" if torch.cuda.is_available() else "cpu")


# In[13]:


st.write("Welcome to Automatic Speech Recognizer System!")


# In[ ]:


def get_params(file):

    file = io.BytesIO(file.read())  # Convert the uploaded file to BytesIO
    data, samplerate = sf.read(file)
    waveform = torch.tensor(data, dtype=torch.float32)
    print(f"File loaded successfully with sample rate: {samplerate}")
    return waveform.unsqueeze(0),int(samplerate)
    

# In[17]:


def model_init():
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    return bundle


# In[19]:


def resampling(file):

    waveform,sample_rate=get_params(file)
    bundle =model_init()
    if sample_rate != model_init().sample_rate:
        waveform = torchaudio.functional.resample(waveform,sample_rate,model_init().sample_rate)
    return waveform


# In[ ]:


def feature_extract_and_predictions(waveform,bundle,file):

    with torch.inference_mode():
        features,_=bundle.get_model().to(device).extract_features(waveform)
        predictions,_ = bundle.get_model().to(device)(waveform)
    return predictions


# In[ ]:


class GreedyCTCDecoder(torch.nn.Module):

    def __init__(self,labels,blank=0):
        super().__init__()
        self.labels=labels
        self.blank=blank
    def forward(self,emissions: torch.Tensor):
        pred=torch.argmax(emissions,dim=-1)
        pred=torch.unique_consecutive(pred)
        pred = [i for i in pred if i!= self.blank]
        str="".join([self.labels[i] for i in pred])
        return str.replace("|"," ")


# In[7]:


file_uploader=st.file_uploader("choose a file",type=["wav"])

if file_uploader is not None:
    st.audio(file_uploader,format="audio/wav")
    bundle = model_init()
    waveform = resampling(file_uploader)
    predictions=feature_extract_and_predictions(waveform,bundle,file_uploader)
    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    transcript = decoder(predictions[0])
    st.write(transcript)


# In[ ]:
