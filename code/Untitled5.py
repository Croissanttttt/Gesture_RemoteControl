#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gtts import gTTS
from playsound import playsound

s = gTTS("Sample Text")
s.save('sample.mp3')
playsound('sample.mp3')

