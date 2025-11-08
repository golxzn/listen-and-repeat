import os
import sys
import time
import datetime
import difflib
import random
import numpy as np
import keyboard
import sounddevice as sd
import soundfile as sf
import pyttsx3
import speech_recognition as sr
from termcolor import colored

def ensure_dir(name: str) -> None:
	os.makedirs(name, exist_ok=True)


voice_id: int = -1
def speak_text(text: str) -> None:
	engine: pyttsx3.Engine = pyttsx3.init()
	global voice_id
	if voice_id == -1:
		voices = engine.getProperty('voices')
		voice_id = random.choice(voices).id

	engine.setProperty('voice', voice_id)
	engine.say(text)
	try:
		engine.runAndWait()
	except Exception as e:
		print(f"[ERROR]: {e}")

def play_beep(freq: float = 880.0, duration: float = 0.8, samplerate: int = 44100) -> None:
	t: np.ndarray = np.linspace(0, duration, int(samplerate * duration), False)
	tone: np.ndarray = 0.5 * np.sin(freq * 2 * np.pi * t)
	sd.play(tone, samplerate)
	sd.wait()

def record_audio(duration: float, samplerate: int = 44100, device_index: int = None) -> np.ndarray:
	print(colored(f"[Recording for {duration:.2f} sec...]", "yellow"))
	rec: np.ndarray = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32', device=device_index)
	for i in range(int(duration * 10)):
		time.sleep(0.1)
		remain: float = duration - i * 0.1
		print(f"\rTime left (space to interrupt): {remain:5.2f} sec", end="")
		if keyboard.is_pressed('space'):
			sd.stop()
			break

	print(f"\rTime left (space to interrupt): 0.00 sec")
	sd.wait()
	return rec

def recognize_offline(wav_path: str) -> str:
	recognizer: sr.Recognizer = sr.Recognizer()
	with sr.AudioFile(wav_path) as src:
		audio_data: sr.AudioData = recognizer.record(src)
	try:
		text: str = recognizer.recognize_sphinx(audio_data)
	except (sr.UnknownValueError, sr.RequestError):
		text = ""
	return text.upper()

def highlight_diff(a: str, b: str) -> str:
	seq: difflib.SequenceMatcher = difflib.SequenceMatcher(None, a, b)
	highlighted: str = ""
	for tag, i1, i2, j1, j2 in seq.get_opcodes():
		if tag == "equal":
			highlighted += a[i1:i2]
		elif tag == "replace":
			highlighted += colored(a[i1:i2], "red", attrs=["bold"])
		elif tag == "delete":
			highlighted += colored(a[i1:i2], "red")
		elif tag == "insert":
			highlighted += colored(b[j1:j2], "green")
	return highlighted

def similarity(a: str, b: str) -> float:
	return difflib.SequenceMatcher(None, a, b).ratio()

def main() -> None:
	if len(sys.argv) != 2:
		print("Usage: python listen-and-repeat.py samples00.txt")
		sys.exit(1)

	sample_file: str = sys.argv[1]
	if not os.path.exists(sample_file):
		print("File not found:", sample_file)
		sys.exit(1)

	samples: list[str] = []
	with open(sample_file, "r", encoding="utf-8") as f:
		samples = f.readlines()

	print(colored("Available audio input devices:", "cyan"))
	devices: list[dict] = sd.query_devices()
	for i, dev in enumerate(devices):
		if dev["max_input_channels"] > 0:
			print(f"[{i}] {dev['name']}")

	device_index: int = -1
	while device_index < 0 or device_index >= len(devices) or devices[device_index]["max_input_channels"] == 0:
		try:
			device_index = int(input(colored("Select microphone index: ", "yellow")))
		except ValueError:
			device_index = -1

	timestamp: str = datetime.datetime.now().strftime("%y%m%d-%H%M")
	base_name: str = os.path.splitext(os.path.basename(sample_file))[0]
	out_dir: str = f"out/{base_name}-{timestamp}"
	ensure_dir(out_dir)

	print(colored("Microphone selected successfully. Starting test", "cyan"))

	scores: list[float] = []
	recognized_texts: list[str] = []
	for idx, sentence in enumerate(samples, start=1):
		print(colored(f"\n[{idx}/{len(samples)}] Sentence #{idx}", "magenta"))
		# print(sentence)

		t0: float = time.time()
		time.sleep(3.0)
		speak_text(sentence)
		time.sleep(1.0)

		play_beep()
		time.sleep(0.5)
		t_elapsed: float = time.time() - t0
		recorded: np.ndarray = record_audio(t_elapsed, device_index=device_index)
		wav_path: str = os.path.join(out_dir, f"sample_{idx:02d}.wav")
		sf.write(wav_path, recorded, 44100)

		recognized_text: str = recognize_offline(wav_path) if os.path.exists(wav_path) else ""
		recognized_texts.append(recognized_text)
		txt_path: str = os.path.join(out_dir, f"sample_{idx:02d}.txt")
		with open(txt_path, "w", encoding="utf-8") as f:
			f.write(sentence.upper() + "\n" + recognized_text + "\n")

		sim: float = similarity(sentence.upper(), recognized_text)
		scores.append(sim)
		# print("Recognized:", colored(recognized_text, "yellow"))
		# print(f"→ Score: {sim*100:.1f}%")

	print("\n" + "=" * 60)
	print(colored("FINAL RESULTS", "green"))
	total: float = float(np.mean(scores) * 100 if scores else 0.0)
	print(f"Average accuracy: {total:.1f}%\n")

	for i in range(len(samples)):
		ref: str = samples[i]
		hyp: str = recognized_texts[i]
		diff: str = highlight_diff(ref.upper(), hyp.upper())
		print(f"{i + 1:02d}. Similarity: {similarity(ref.upper(), hyp.upper())*100:.1f}%")
		print(f"     Reference : {ref}", end="")
		print(f"     Recorded  : {hyp}")
		print(f"     Difference: {diff}")

if __name__ == "__main__":
	main()

