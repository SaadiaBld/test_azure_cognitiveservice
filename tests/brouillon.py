from pydub import AudioSegment

def convert_m4a_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")

# convertir le fichier audio m4a en wav
convert_m4a_to_wav("bonjour.m4a", "bonjour.wav")