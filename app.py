import sys
import multiprocessing
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
        self.root.geometry("620x500")
        
        # Configurazione Stile coerente per tema Dark/Light su Mac
        style = ttk.Style()
        style.theme_use('clam')

        # Titolo e Stato
        self.label_status = tk.Label(root, text="Stato: Pronto", font=("Helvetica", 12, "bold"))
        self.label_status.pack(pady=8)
        
        # Selezione Dispositivo Audio
        frame_audio = tk.Frame(root)
        frame_audio.pack(pady=5)
        tk.Label(frame_audio, text="Ingresso Audio:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.devices = sd.query_devices()
        self.input_devices = [d['name'] for d in self.devices if d['max_input_channels'] > 0]
        
        self.selected_device = tk.StringVar(root)
        if self.input_devices:
            self.selected_device.set(self.input_devices[0])
            
        self.dropdown = ttk.OptionMenu(frame_audio, self.selected_device, self.selected_device.get(), *self.input_devices)
        self.dropdown.pack(side=tk.LEFT, padx=5)
        
        # Area di testo Trascrizione
        self.text = scrolledtext.ScrolledText(root, font=("Menlo", 11), bg="#1e1e1e", fg="#ffffff", insertbackground="white")
        self.text.pack(expand=True, fill='both', padx=15, pady=10)
        
        # Pulsante Avvia / Ferma con stile visibile
        self.btn = tk.Button(
            root, 
            text="AVVIA TRASCRIZIONE", 
            font=("Helvetica", 11, "bold"), 
            command=self.toggle, 
            bg="#28a745", 
            fg="white", 
            activebackground="#218838", 
            activeforeground="white",
            relief="raised",
            bd=2,
            padx=15,
            pady=5
        )
        self.btn.pack(pady=12)
        
        self.running = False
        self.model = None
        self.audio_queue = queue.Queue()
        self.stream = None

    def get_device_index(self, name):
        for i, d in enumerate(self.devices):
            if d['name'] == name:
                return i
        return None

    def toggle(self):
        if not self.running:
            self.start_transcription()
        else:
            self.stop_transcription()

    def start_transcription(self):
        if not self.model:
            self.label_status.config(text="Caricamento modello Whisper in corso...", fg="orange")
            self.root.update()
            # Carica modello locale
            self.model = WhisperModel("base", device="cpu", compute_type="int8")
            
        self.running = True
        self.btn.config(text="FERMA TRASCRIZIONE", bg="#dc3545", activebackground="#c82333", fg="white")
        self.label_status.config(text="Stato: In ascolto...", fg="#28a745")
        
        # Svuota la coda
        while not self.audio_queue.empty():
            self.audio_queue.get()

        threading.Thread(target=self.loop, daemon=True).start()

    def stop_transcription(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.btn.config(text="AVVIA TRASCRIZIONE", bg="#28a745", activebackground="#218838", fg="white")
        self.label_status.config(text="Stato: Fermato", fg="orange")

    def loop(self):
        device_name = self.selected_device.get()
        device_idx = self.get_device_index(device_name)
        
        def callback(indata, frames, time, status):
            if self.running:
                self.audio_queue.put(indata.copy())

        try:
            self.stream = sd.InputStream(samplerate=16000, channels=1, device=device_idx, callback=callback)
            with self.stream:
                while self.running:
                    audio_data = []
                    # Accumula 3 secondi di audio
                    for _ in range(int(16000 * 3 / 1024)):
                        if not self.running:
                            break
                        try:
                            audio_data.append(self.audio_queue.get(timeout=0.5))
                        except queue.Empty:
                            continue

                    if audio_data and self.running:
                        np_data = np.concatenate(audio_data, axis=0).flatten().astype(np.float32)
                        segments, _ = self.model.transcribe(np_data, language="it", beam_size=1)
                        for seg in segments:
                            if seg.text.strip():
                                text_to_insert = seg.text.strip() + "\n"
                                self.root.after(0, self._append_text, text_to_insert)
        except Exception as e:
            self.root.after(0, self._handle_error, str(e))

    def _append_text(self, text):
        self.text.insert(tk.END, text)
        self.text.see(tk.END)

    def _handle_error(self, err_msg):
        self.text.insert(tk.END, f"\n[Errore Audio]: {err_msg}\n")
        self.stop_transcription()
        self.label_status.config(text="Errore dispositivo audio", fg="red")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    App(root)
    root.mainloop()
