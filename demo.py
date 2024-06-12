#construire une demo avec gradio 
import gradio as gr
import asyncio
import shutil
import tempfile
from chaine import transcription_audio_texte, analyse_sentiment_enregistrement, if_negative_sentiment

async def process_audio(audio):
    try: 
        transcription = await transcription_audio_texte(audio)
        if audio is None or not hasattr(audio, 'name'):
            raise ValueError("Invalid audio input")
    
        #le fichier audio enregistré par gradio est stocké temporairement, pour qu'il soit persistant et disponible pour la fonction de transciption, on le copie dans un fichier temporaire 
        audio_path = tempfile.mktemp(suffix=".wav")
        shutil.copyfile(audio.name, audio_path)
        #transcription de l'audio
        transcription = await transcription_audio_texte(audio_path)

        #analyse de sentiment
        sentiment = analyse_sentiment_enregistrement(transcription)

        return transcription, sentiment
    
    except Exception as e:
        return "Error during processing: " + str(e), None

#cree une interface gradio pour la demo, l'interface utilise les fonction definies dans chaine.py 
# demotest = gr.Interface(fn=process_audio, inputs=gr.Audio(sources=['microphone'], type='filepath'),
#                         outputs = [gr.Textbox(label="Transcription"), gr.Textbox(label="Sentiment")])

demotest = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Upload audio file"), 
    outputs=[gr.Textbox(label="Transcription"), gr.Textbox(label="Sentiment")]
)
demotest.launch()


