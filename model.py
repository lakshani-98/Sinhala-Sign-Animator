from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import fasttext
import nltk
import string
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from google.cloud import translate_v2 as translate
from IPython.display import Video, display
import joblib
from moviepy.editor import VideoFileClip, concatenate_videoclips

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

keyPath = ''

rf = joblib.load(keyPath+'rf.pkl') 

model = fasttext.load_model(keyPath+'cc.en.300.bin')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyPath+'translate-key.json' 

video_folder_path = keyPath+'Signs/'

stopwords = {'the', 'a', 'an','is', 'are', 'was', 'were', 'am', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
             'would', 'shall', 'should', 'may', 'might', 'must', }

def get_embedding(word):
    return model.get_word_vector(word)


#Remove stop words
def preprocess_sentence(sentence):
    sentence = sentence.lower()
    words = nltk.word_tokenize(sentence)

    filtered_words = [word for word in words if word not in stopwords]

    filtered_sentence = ' '.join(filtered_words)

    return filtered_sentence


def move_verb_to_end(sentence):
    words = nltk.word_tokenize(sentence)

    pos_tags = nltk.pos_tag(words)

    verb = None
    verb_index = None
    for i, (word, tag) in enumerate(pos_tags):
        if tag.startswith('VB'):  # Check if the tag is a verb (VB, VBD, VBG, VBN, VBP, VBZ)
            verb = word
            verb_index = i
            break

    if verb is not None:
        words.pop(verb_index)
        words.append(verb)

    modified_sentence = ' '.join(words)

    return modified_sentence

def swap_preposition_noun_pairs(sentence):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)

    preposition_noun_pairs = []

    # Identify preposition-noun pairs
    i = 0
    while i < len(pos_tags):
        word, tag = pos_tags[i]
        
        if tag == 'IN' and i + 1 < len(pos_tags):
            next_word, next_tag = pos_tags[i + 1]
            
            if next_word in ['a', 'an', 'the']:
                if i + 2 < len(pos_tags):
                    next_word, next_tag = pos_tags[i + 2]
                    if next_tag.startswith('NN'):
                        preposition_noun_pairs.append((i, i + 2))
                i += 2  
            elif next_tag.startswith('NN'):
                preposition_noun_pairs.append((i, i + 1))
                i += 1  
        i += 1

    # Swap prepositions and the following nouns
    for index, next_index in reversed(preposition_noun_pairs):
        if index < len(words) and next_index < len(words):
            words[index], words[next_index] = words[next_index], words[index]

    modified_sentence = ' '.join(words)
    return modified_sentence


def translate_text(text, target_language='en'):
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

def remove_punctuation(sentence):
    translator = str.maketrans('', '', string.punctuation)

    cleaned_sentence = sentence.translate(translator)

    return cleaned_sentence


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def count_files_in_directory(directory):
    count = 0
    for _, _, files in os.walk(directory):
        count += len(files)
    return count

def generate_sign_language_video(sign_keys_arrays):
    sign_keys = flatten_list(sign_keys_arrays)
    video_files = [video_folder_path +  key + '.mkv' for key in sign_keys]
    valid_clips = []

    for video_file in video_files:
        if os.path.exists(video_file):
            try:
                clip = VideoFileClip(video_file)
                valid_clips.append(clip)
            except Exception as e:
                print(f"Error loading video {video_file}: {e}")
        else:
            print(f"Video file {video_file} does not exist")

    if not valid_clips:
        raise ValueError("No valid video clips found")

    # Concatenate the valid clips
    final_clip = concatenate_videoclips(valid_clips)

    directory = keyPath+'output/' 
    file_count = count_files_in_directory(directory) + 1
    video_name = f"output_{file_count}.mp4"
    output_video_path = f"{keyPath}output/output_{file_count}.mp4"
    
    final_clip.write_videofile(output_video_path)
    
    return video_name

def preprocess_and_predict(sinhalaText):
  engText = translate_text(sinhalaText)
  print(engText)
  sentences = nltk.sent_tokenize(engText)
  processed_sentences = []
  for sentence in sentences:
      sentence = move_verb_to_end(sentence)
      sentence = swap_preposition_noun_pairs(sentence)
      sentence = preprocess_sentence(sentence)
      sentence = remove_punctuation(sentence)
      processed_sentences.append(sentence)

  # Split each processed sentence into words and get embeddings
  predictions = []
  for sentence in processed_sentences:
      words = sentence.split()
      word_embeddings = [get_embedding(word) for word in words]

      # Convert embeddings to DataFrame
      embeddings_df = pd.DataFrame(word_embeddings)

      # Predict the sign for each word's embedding
      sentence_predictions = rf.predict(embeddings_df)
      predictions.append(sentence_predictions)
  
  print(predictions)

  video_name = generate_sign_language_video(predictions)

  return video_name