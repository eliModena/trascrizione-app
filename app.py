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
        
        style = ttk.Style()
        style.theme_use('clam')

        self.label_status = tk.Label(root, text="Stato: Pronto", font=("Helvetica", 12, "bold"))
        self.label_status.pack(pady=8)
        
        frame_audio = tk.Frame(root)
        frame_audio.pack(pady=5)
        tk.Label(frame_audio, text="Ingresso Audio:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        try:
            self.devices = sd.query_devices()
            self.input_devices = [d['name'] for d in self.devices if d['max_input_channels'] > 0]
        except Exception as e:
            self.input_devices = []

        self.selected_device = tk.StringVar(root)
        if self.input_devices:
            self.selected_device.set(self.input_devices[0])
            
        self.dropdown = ttk.OptionMenu(frame_audio, self.selected_device, self.selected_device.get() if self.input_devices else "Nessun microfono", *self.input_devices if self.input_devices else ["Nessun microfono"])
        self.dropdown.pack(side=tk.LEFT, padx=5)
        
        self.text = scrolledtext.ScrolledText(root, font=("Menlo", 11), bg="#1e1e1e", fg="#ffffff", insertbackground="white")
        self.text.pack(expand=True, fill='both', padx=15, pady=10)
        
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
            self.model = WhisperModel("base", device="cpu", compute_type="int8")
            
        self.running = True
        self.btn.config(text="FERMA TRASCRIZIONE", bg="#dc3545", activebackground="#c82333", fg="white")
        self.label_status.config(text="Stato: In ascolto...", fg="#28a745")
        
        threading.Thread(target=self.loop, daemon=True).start()

    def stop_transcription(self):
        self.running = False
        self.btn.config(text="AVVIA TRASCRIZIONE", bg="#28a745", activebackground="#218838", fg="white")
        self.label_status.config(text="Stato: Fermato", fg="orange")

    def loop(self):
        device_name = self.selected_device.get()
        device_idx = self.get_device_index(device_name)
        fs = 16000
        block_duration = 3  # secondi per blocco audio
        frames_per_block = int(fs * block_duration)

        self.root.after(0, self._append_text, f">> Avvio cattura audio su: {device_name}\n")

        try:
            while self.running:
                # Registrazione bloccante a blocchi (permette a macOS di tracciare l'accesso hardware)
                audio_data = sd.rec(frames_per_block, samplerate=fs, channels=1, dtype='float32', device=device_idx)
                sd.wait()  # Attende la fine della registrazione del blocco
                
                if not self.running:
                    break

                np_data = audio_data.flatten()
                volume = np.linalg.norm(np_data)
                
                if volume < 0.001:
                    self.root.after(0, self._append_text, "[Debug]: Silenzio rilevato dal microfono\n")
                else:
                    self.root.after(0, self._append_text, f"[Debug]: Audio attivo! (Vol: {volume:.4f})\n")

                segments, _ = self.model.transcribe(np_data, language="it", beam_size=1)
                for seg in segments:
                    if seg.text.strip():
                        self.root.after(0, self._append_text, seg.text.strip() + "\n")
                        
        except Exception as e:
            self.root.after(0, self._append_text, f"\n[Errore Critico Audio]: {str(e)}\n")
            self.stop_transcription()

    def _append_text(self, text):
        self.text.insert(tk.END, text)
        self.text.see(tk.END)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    App(root)
    root.mainloop()
