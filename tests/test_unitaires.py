import pytest
from unittest.mock import patch, MagicMock
from azure.cognitiveservices.speech import SpeechRecognizer, SpeechConfig, audio, ResultReason
from process_audio_functions import transcription_audio_texte, enregistrer_texte_mongodb, analyse_sentiment, if_negative_sentiment
from dotenv import load_dotenv
import sys
import os

#ajouter repertoire parent au path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

def test_transcription_audio_texte_valid():
    with patch('azure.cognitiveservices.speech.SpeechRecognizer.recognize_once') as mock_recognize_once: #la fonction recognize once est pacthée c'est à dire qu'elle est remplacée par un mock pour retourner un résultat fixe
        mock_result = MagicMock() #simule résultat de la reconnaissance vocale
        mock_result.reason = ResultReason.RecognizedSpeech
        mock_result.text = "Bonjour"
        mock_recognize_once.return_value = mock_result
        
        audio_path = "tests/bonjour.wav"
        result = transcription_audio_texte(audio_path) #appel de la fonction à tester
        assert result == "Bonjour"

def test_transcription_audio_texte_no_match():
    with patch('azure.cognitiveservices.speech.SpeechRecognizer.recognize_once') as mock_recognize_once:
        mock_result = MagicMock()
        mock_result.reason = ResultReason.NoMatch
        mock_recognize_once.return_value = mock_result
        
        audio_path = "tests/bonjour.wav"
        result = transcription_audio_texte(audio_path)
        assert result == "Echec de la reconnaissance vocale"
        
def test_transcription_audio_texte_invalid_file():
    result = transcription_audio_texte("tests/audio.wav")
    assert result == "Fichier audio non valide"

def test_transcription_audio_texte_no_audio():
    result = transcription_audio_texte(None)
    assert result == "Aucun fichier audio fourni"