import pyaudio
import wave

CHUNK = 4096  # 進一步增加帧大小
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p =pyaudio.PyAudio()
stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)
print("Start recording...")

frames = []
seconds = 3

for i in range(0,int(RATE/CHUNK*seconds)):
	data = stream.read(CHUNK)
	frames.append(data)
print("reconding stopped")

stream.stop_stream()
stream.close()

wf=wave.open("output.wav",'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()