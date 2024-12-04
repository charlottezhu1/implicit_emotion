"""
This python file runs a portion of the tweet json file base on the unique tweet id in the group.
Input: json file for tweet; group id
Output: csv with emotions. 
"""
import json
import re
import random
from openai import OpenAI
import csv
import os
from dotenv import load_dotenv
import time
import argparse
from argparse import RawTextHelpFormatter

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), organization='org-LB0f6h3G5argFODGs1DOSLNn')



import json


def gpt_detect_emotion_category(tweet_text):
    """
    Calls GPT to determine the specific emotion type and associated arousal levels in the input tweet.
    
    - This function focuses on categorizing the tweet's emotional tone based on arousal and sentiment.
    - The temperature is optimized for balanced response quality based on experiments with sample outputs.
    
    Parameters:
    - tweet_text (str): The text of the tweet to analyze.
    
    Returns:
    - dict: A dictionary with emotion categories and a short explanation.
    """
    
    # Define the expected output schema
    schema = {
        "Neutral": "<0_or_1>",
        "Low-Arousal Positive": "<0_or_1>",
        "High-Arousal Positive": "<0_or_1>",
        "Low-Arousal Negative": "<0_or_1>",
        "High-Arousal Negative": "<0_or_1>",
        "explanation": "This tweet would make the user feel <category_placeholder> because..."
    }
    
    # Construct the prompt for GPT
    prompt = f"""
    As an expert annotator specializing in emotions in social media content, your job is to predict the emotions and feelings that Twitter messages might evoke in users.

    Emotions and arousal levels can be categorized as follows:

    - **Neutral**: No strong positive or negative emotion detected. 
    - **Low-Arousal Positive**: Peaceful, calm, relaxed. 
    - **High-Arousal Positive**: Enthusiastic, excited, elated. 
    - **Low-Arousal Negative**: Sluggish, sleepy, dull. 
    - **High-Arousal Negative**: Fearful, hostile, nervous. 

    Think about the emotion the tweet might cause the user to feel, and choose the corresponding emotion and arousal category.
    Assign 1 if it applies; assign 0 otherwise. Choose only one out of the five categories.

    Input: {tweet_text}

    Output format: JSON object matching this structure:
    {json.dumps(schema, indent=2)}

    Example input:
    "I’m passionate about indie app development because I’ve been able to take months off at a time for my health and have no impact on my income 🩷 this should be available to way more people"

    Example output:
    {{
        "Neutral": 0,
        "Low-Arousal Positive": 0,
        "High-Arousal Positive": 1,
        "Low-Arousal Negative": 0,
        "High-Arousal Negative": 0,
        "explanation": "This tweet would make the user feel excited and joyful, which falls under the high-arousal positive category. This is because the tweet expresses strong enthusiasm and satisfaction with the flexibility of indie app development, highlighted by the word 'passionate' and the heart emoji."
    }}
    """

    
    # Make the GPT API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Provide output in valid JSON with this schema:" + json.dumps(schema)},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.6,
    )
    
    # Parse and return the JSON response
    data = json.loads(response.choices[0].message.content)
    # print("Detected Emotion Type:", data)
    return data


def detect_tweet_emotion(filename): 
    """
    Repeatedly calls GPT to evaluate the emotional tone of each tweet in the given file
    and stores the results in a new CSV file.

    Parameters:
    - filename (str): The name of the input CSV file containing tweets.

    Output:
    - A CSV file (output_<filename>.csv) with the tweet emotions categorized.
    """

    gpt_header = [
        "id",
        "tweet",
        "Neutral",
        "Low-Arousal Positive",
        "High-Arousal Positive",
        "Low-Arousal Negative",
        "High-Arousal Negative",
        "explanation"
    ]
    
    call_count = 0
  
    # Open the output file and write the header
    with open(f'./output_{filename}.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(gpt_header)

        # Read the input file and skip the header row
        with open(f'./{filename}', 'r') as input_file:
            reader = csv.DictReader(input_file)
            
            for row in reader:
                tweet_id = row["tweet_id"]
                tweet = row["tweet"]

                gpt_result = gpt_detect_emotion_category(tweet)  
                
                gpt_row = [
                    tweet_id,
                    tweet,
                    gpt_result["Neutral"],
                    gpt_result["Low-Arousal Positive"],
                    gpt_result["High-Arousal Positive"],
                    gpt_result["Low-Arousal Negative"],
                    gpt_result["High-Arousal Negative"],
                    gpt_result["explanation"]
                ]
                writer.writerow(gpt_row)
                
                call_count += 1

                # Print status every 5 tweets
                if call_count % 5 == 0:
                    print("Finished processing tweet number:", call_count)

                # Pause every 1000 API calls to manage rate limits
                if call_count % 1000 == 0:
                    print("Pausing for 5 seconds to avoid rate limits... Current Count:", call_count)
                    time.sleep(5)     


def main(): 
  parser = argparse.ArgumentParser(
        description="""Run the emotion detection on a txt file containing unique tweet ids.
Run it like: python run_server.py "filename" """,
        formatter_class=RawTextHelpFormatter
    )
  parser.add_argument("filename", type=str, help="The filename that contains unique tweet IDs to run the script on.")
  args = parser.parse_args()
  detect_tweet_emotion(args.filename)


if __name__ == '__main__':
  main()