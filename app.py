import threading
import queue
import tkinter as tk
from tkinter import scrolledtext, ttk
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("BE Video Intercom Transcriptor")
        self.root.geometry("600x450")
        
        # Etichetta di stato
        self.label = tk.Label(root, text="Pronto", font=("Arial", 12, "bold"))
        self.label.pack(pady=5)
        
        # Selezione Dispositivo Audio
        frame_audio = tk.Frame(root)
        frame_audio.pack(pady=5)
        tk.Label(frame_audio, text="Dispositivo Audio:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        self.devices = sd.query_devices()
        self.input_devices = [d['name'] for d in self.devices if d['max_input_channels'] > 0]
        
        self.selected_device = tk.StringVar(root)
        if self.input_devices:
            self.selected_device.set(self.input_devices[0])
        
        self.dropdown = tk.OptionMenu(frame_audio, self.selected_device, *self.input_devices)
        self.dropdown.config(font=("Arial", 10))
        self.dropdown.pack(side=tk.LEFT, padx=5)
        
        # Area di testo per la trascrizione
        self.text = scrolledtext.ScrolledText(root, font=("Arial", 10))
        self.text.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Pulsante Avvia / Ferma (usiamo foreground esplicito per evitare problemi di visibilità)
        self.btn = tk.Button(root, text="Avvia", font=("Arial", 11, "bold"), command=self.toggle, bg="#28a745", fg="white", activebackground="#218838", activeforeground="white")
        self.btn.pack(pady=10)
        
        self.running = False
        self.model = None
        self.audio_queue = queue.Q() if hasattr(queue, 'Q') else queue.Queue()

    def get_device_index(self, name):
        for i, d in enumerate(self.devices):
            if d['name'] == name:
                return i
        return None

    def toggle(self):
        if not self.running:
            if not self.model:
                self.label.config(text="Caricamento modello in corso...", fg="orange")
                self.root.update()
                self.model = WhisperModel("base", device="cpu", compute_type="int8")
            
            self.running = True
            self.btn.config(text="Ferma", bg="#dc3545", activebackground="#c82333")
            self.label.config(text="In ascolto...", fg="green")
            
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()
        else:
            self.running = False
            self.btn.config(text="Avvia", bg="#28a745", activebackground="#218838")
            self.label.config(text="Fermato", fg="orange")

    def loop(self):
        device_name = self.selected_device.get()
        device_idx = self.get_device_index(device_name)
        
        q = queue.Queue()
        def callback(indata, frames, time, status):
            if self.running:
                q.put(indata.copy())

        try:
            with sd.InputStream(samplerate=16000, channels=1, device=device_idx, callback=callback):
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
        except Exception as e:
            self.text.insert(tk.END, f"\n[Errore audio]: {e}\n")
            self.running = False
            self.btn.config(text="Avvia", bg="#28a745")
            self.label.config(text="Errore", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
