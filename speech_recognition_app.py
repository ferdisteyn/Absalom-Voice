import os
import signal
import pyaudio
import warnings
import threading
import numpy as np
import tkinter as tk
import soundfile as sf
from tkinter import ttk
import sounddevice as sd
import matplotlib.pyplot as plt
from transformers import pipeline
from scipy.signal import butter, lfilter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

pipe = pipeline(model="CMPG313/absalom_voice_model")  # change to "your-username/the-name-you-picked"

def record_audio(duration, filename):
    sample_rate = 44100
    channels = 2

    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    sf.write(file_path, recording, sample_rate)

def transcribe(audio_file):
    with open(audio_file, "rb") as f:
        audio = f.read()
    text = pipe(audio)["text"]
    return text

def record_and_transcribe():
    for i in range(0, 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clip_filename = f"clip{i}.wav"
            myOutput.insert(tk.END,"Start speaking...\n")
            threading.Thread(target=visualize_audio).start()
            record_audio(10, clip_filename)
            threading.Thread(target=stop_visualize_audio).start()
            myOutput.insert(tk.END,"Stop speaking.\n")
            text = transcribe(os.path.dirname(os.path.abspath(__file__))+"/"+clip_filename)
            myOutput.insert(tk.END, "You said: " + text.strip() + "\n")
    output = text
    return output

def visualize_audio():
    while streaming.get():
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16)
        samples = samples / 32768.0
        wave = np.sin(np.arange(CHUNK_SIZE) * (1 * np.pi * 1 / CHUNK_SIZE))
        modulated_wave = samples * wave
        filtered_wave = lfilter(b, a, modulated_wave)
        line.set_ydata(filtered_wave)
        fig.canvas.draw()
        plt.pause(0.001)

def stop_visualize_audio():
    global streaming
    streaming.set(False)

def start_streaming():
    global streaming
    streaming.set(True)
    threading.Thread(target=record_and_transcribe).start()

def close():
    window.destroy()
    os.kill(os.getpid(), signal.SIGINT)


################################################## GUI CODE ##################################################


CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                input_device_index=sd.default.device['input'])

fig = plt.Figure(facecolor='#2f2f2f', figsize=(2, 2))

ax = fig.add_subplot(111)

x = np.arange(0, CHUNK_SIZE)
line, = ax.plot(x, np.zeros(CHUNK_SIZE), color='#aaaaaa')

ax.set_ylim(-1, 1)
ax.set_xlim(0, CHUNK_SIZE)
ax.axis('off')

nyq = 0.5 * RATE
cutoff = 500
order = 5
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)

# Create the main window
window = tk.Tk()
window.title("Speech Recognition App")
window.resizable(width=False, height=False)
window.geometry("649x277")
window.config(bg="#2f2f2f")
window.protocol("WM_DELETE_WINDOW", close)

top_frame = ttk.Frame(window, width=600, height=12, padding=10, style="DarkFrame.TFrame").grid(row=0, columnspan=2)
middle_frame = ttk.Frame(window, width=600, height=190, padding=10, style="LightFrame.TFrame").grid(row=1, columnspan=2)
bottom_frame = ttk.Frame(window, width=600, height=10, padding=19, style="DarkFrame.TFrame").grid(row=2, columnspan=2)

myOutput = tk.Text(master=middle_frame, wrap="word", height=12, width=54, background="#2f2f2f", foreground="white", border=1)
myOutput.grid(row=1, column=1, columnspan=2)
canvas = FigureCanvasTkAgg(fig, master=middle_frame)
canvas.get_tk_widget().grid(row=1) 

streaming = tk.BooleanVar()
streaming.set(False)

buttonStart = ttk.Button(master=bottom_frame, text='Record & Transcribe', width=18, command=start_streaming, style="RoundButton.TButton")
buttonStart.grid(row=3, column=0)

style = ttk.Style()
style.configure("DarkFrame.TFrame", background="#2f2f2f")
style.configure("LightFrame.TFrame", background="#2f2f2f")
style.configure("RoundButton.TButton", foreground="black", background="#2f2f2f", borderwidth=4, focuscolor="#red", font=("Arial", 8), padding=10, relief="flat", borderradius=20)


# Start the main event loop
window.mainloop()

stream.stop_stream()
stream.close()
p.terminate()