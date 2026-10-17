import threading, queue
import tkinter as tk
from tkinter import scrolledtext
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Trascrizione Offline")
        self.root.geometry("500x300")
        self.label = tk.Label(root, text="Pronto", font=("Arial", 12, "bold"))
        self.label.pack(pady=10)
        self.text = scrolledtext.ScrolledText(root, font=("Arial", 10))
        self.text.pack(expand=True, fill='both', padx=10, pady=10)
        self.btn = tk.Button(root, text="Avvia", command=self.toggle, bg="green", fg="white")
        self.btn.pack(pady=10)
        self.running = False
        self.model = None

    def toggle(self):
        if not self.running:
            if not self.model:
                self.label.config(text="Caricamento modello...", fg="orange")
                self.root.update()
                self.model = WhisperModel("base", device="cpu", compute_type="int8")
            self.running = True
            self.btn.config(text="Ferma", bg="red")
            self.label.config(text="In ascolto...", fg="green")
            threading.Thread(target=self.loop, daemon=True).start()
        else:
            self.running = False
            self.btn.config(text="Avvia", bg="green")
            self.label.config(text="Fermato", fg="orange")

    def loop(self):
        with sd.InputStream(samplerate=16000, channels=1, callback=lambda indata, f, t, s: q.put(indata.copy())):
            global q
            q = queue.Queue()
            while self.running:
                try:
                    data = [q.get(timeout=1)]
                    np_data = np.concatenate(data, axis=0).flatten().astype(np.float32)
                    segments, _ = self.model.transcribe(np_data, language="it", beam_size=1)
                    for seg in segments:
                        if seg.text.strip():
                            self.text.insert(tk.END, seg.text.strip() + "\n")
                            self.text.see(tk.END)
                except queue.Empty:
                    continue

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
