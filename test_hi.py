from gtts import gTTS
tts = gTTS('नमस्ते, यह एक परीक्षण ऑडियो है।', lang='hi')
tts.save('test_hi.mp3')
print('Saved test_hi.mp3')
