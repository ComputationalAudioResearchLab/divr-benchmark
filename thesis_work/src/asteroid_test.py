import librosa
import soundfile
import asteroid

model = asteroid.models.BaseModel.from_pretrained(
    "mpariente/DPRNNTasNet-ks2_WHAM_sepclean"
)

sample_rate = 16000
audio_file_path = "/home/storage/divr_benchmark/data/voiced/voice-icar-federico-ii-database-1.0.0/voice002.wav"
audio, _ = librosa.load(audio_file_path, sr=sample_rate, mono=True)
print(audio.shape)

separated = model.separate(audio[None, None, :])[0]
print(separated.shape)

soundfile.write("first.wav", data=separated[0], samplerate=sample_rate)
soundfile.write("second.wav", data=separated[1], samplerate=sample_rate)
