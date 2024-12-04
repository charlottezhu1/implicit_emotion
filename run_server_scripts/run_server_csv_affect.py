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


def gpt_detect_emotion_category(tweet_text):
    """
    Call GPT to determine the specific emotion type in the input tweet.
    """
    schema = {
        "Nervous": "<0_or_1>",
        "Sad": "<0_or_1>",
        "Happy": "<0_or_1>",
        "Calm": "<0_or_1>",
        "Excited": "<0_or_1>",
        "Aroused": "<0_or_1>",
        "Angry": "<0_or_1>",
        "Relaxed": "<0_or_1>",
        "Fearful": "<0_or_1>",
        "Enthusiastic": "<0_or_1>",
        "Still": "<0_or_1>",
        "Satisfied": "<0_or_1>",
        "Bored": "<0_or_1>",
        "Lonely": "<0_or_1>",
        "Tired": "<0_or_1>",
        "explanation": "The tweet is likely to cause the user to feel <emotion_placeholder> because..."
    }

    prompt = f"""
    As an expert annotator specializing in emotions in social media content, your job is to predict the emotions and feelings that Twitter messages might evoke in users. Please identify the emotions the input could cause the user to feel. Assign a 1 to the emotion category if it applies; otherwise, assign a 0.

    Choose from the following emotions:
    - Nervous
    - Sad
    - Happy
    - Calm
    - Excited
    - Aroused
    - Angry
    - Relaxed
    - Fearful
    - Enthusiastic
    - Still
    - Satisfied
    - Bored
    - Lonely
    - Tired

    If none of them apply, return 0 for all. Also, provide a brief explanation. 

    Input: {tweet_text} 

    Output format: JSON object matching this structure:
    {json.dumps(schema, indent=2)}

    Example input:
    "I'm passionate about indie app development because I've been able to take months off at a time for my health and have no impact on my income 🩷"

    Example output:
    {{
        "Nervous": 0,
        "Sad": 0,
        "Happy": 1,
        "Calm": 0,
        "Excited": 0,
        "Aroused": 0,
        "Angry": 0,
        "Relaxed": 0,
        "Fearful": 0,
        "Enthusiastic": 1,
        "Still": 0,
        "Satisfied": 1,
        "Bored": 0,
        "Lonely": 0,
        "Tired": 0,
        "explanation": "The tweet is likely to cause the user to feel satisfied, happy, and enthusiastic because it conveys fulfillment with the flexible, financially stable lifestyle of indie app development and passionate excitement, highlighted by the heart emoji 🩷."
    }}
    """

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

    data = json.loads(response.choices[0].message.content)
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
        "Nervous",
        "Sad",
        "Happy",
        "Calm",
        "Excited",
        "Aroused",
        "Angry",
        "Relaxed",
        "Fearful",
        "Enthusiastic",
        "Still",
        "Satisfied",
        "Bored",
        "Lonely",
        "Tired",
        "explanation"
    ]

    
    call_count = 0
  
    # Open the output file and write the header
    with open(f'./output_affect_{filename}.csv', 'w', newline='') as output_file:
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
                    gpt_result["Nervous"],
                    gpt_result["Sad"],
                    gpt_result["Happy"],
                    gpt_result["Calm"],
                    gpt_result["Excited"],
                    gpt_result["Aroused"],
                    gpt_result["Angry"],
                    gpt_result["Relaxed"],
                    gpt_result["Fearful"],
                    gpt_result["Enthusiastic"],
                    gpt_result["Still"],
                    gpt_result["Satisfied"],
                    gpt_result["Bored"],
                    gpt_result["Lonely"],
                    gpt_result["Tired"],
                    gpt_result["explanation"]
                ]

                writer.writerow(gpt_row)
                
                call_count += 1

                # Print status every 5 tweets
                if call_count % 10 == 0:
                    print("Finished processing tweet number:", call_count)

                # Pause every 1000 API calls to manage rate limits
                if call_count % 50 == 0:
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