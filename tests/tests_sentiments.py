import pytest
from unittest.mock import patch, MagicMock
from azure.cognitiveservices.speech import SpeechRecognizer, SpeechConfig, audio, ResultReason
from azure.ai.textanalytics import TextAnalyticsClient
from process_audio_functions import transcription_audio_texte, enregistrer_texte_mongodb, analyse_sentiment, if_negative_sentiment
from dotenv import load_dotenv
import sys
import os

#Cas positif pour un texte positif
def test_analyse_sentiment_positive():
    with patch('azure.ai.textanalytics.TextAnalyticsClient.analyze_sentiment') as mock_analyze_sentiment:
        mock_response = MagicMock()
        mock_doc = MagicMock()
        mock_doc.sentiment = "positive"
        mock_doc.confidence_scores = {"positive": 0.95, "neutral": 0.05, "negative": 0.0}
        mock_response.__getitem__.return_value = mock_doc
        mock_analyze_sentiment.return_value = [mock_doc]
        
        text = "C'est une belle journée"
        sentiment, scores = analyse_sentiment(text)
        assert sentiment == "positive"
        assert scores == {"positive": 0.95, "neutral": 0.05, "negative": 0.0}


#cas pour un texte négatif
def test_analyse_sentiment_negative():
    with patch('azure.ai.textanalytics.TextAnalyticsClient.analyze_sentiment') as mock_analyze_sentiment:
        mock_response = MagicMock()
        mock_doc = MagicMock()
        mock_doc.sentiment = "negative"
        mock_doc.confidence_scores = {"positive": 0.0, "neutral": 0.0, "negative": 1.0}
        mock_response.__getitem__.return_value = mock_doc
        mock_analyze_sentiment.return_value = [mock_doc]

        text = "Les frais de découvert ont triplé le mois dernier."
        sentiment, scores = analyse_sentiment(text)

        assert sentiment == "negative"
        assert scores == {"positive": 0.0, "neutral": 0.0, "negative": 1.0}

#cas pour un texte neutre
def test_analyse_sentiment_neutral():
    with patch('azure.ai.textanalytics.TextAnalyticsClient.analyze_sentiment') as mock_analyze_sentiment:
        mock_response = MagicMock()
        mock_doc = MagicMock()
        mock_doc.sentiment = "neutral"
        mock_doc.confidence_scores = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
        mock_response.__getitem__.return_value = mock_doc
        mock_analyze_sentiment.return_value = [mock_doc]

        text = "J'ai reçu ma carte de retrait aujourd'hui."
        sentiment, scores = analyse_sentiment(text)

        assert sentiment == "neutral"
        assert scores == {"positive": 0.0, "neutral": 1.0, "negative": 0.0}