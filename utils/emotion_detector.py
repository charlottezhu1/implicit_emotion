"""
Emotion Detection Utility - Enhanced Version

This module provides functions to predict emotional impacts of tweets using GPT API.
Supports options for including images, personalized prompts, and user demographic context.

USAGE EXAMPLES:

Basic usage:
    result = gpt_detect_emotion("Great news about the economy!")

With user context (recommended for personalized analysis):
    result = gpt_detect_emotion("Great news about the economy!", participant_id="iP009a45e51df9d4f5")

Complete pipeline (tweet ID + user ID):
    result = detect_emotion_by_tweet_id("tweet-123", participant_id="iP009a45e51df9d4f5")

The system automatically handles user profile lookup, context integration, and personalized prompting.
"""

import json
import os
import pandas as pd
from typing import Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    organization=os.environ.get("OPENAI_ORG_ID", "org-LB0f6h3G5argFODGs1DOSLNn"),
)


def gpt_detect_emotion(
    tweet_text: str,
    media_url: Optional[str] = None,
    include_image: bool = False,
    personalized: bool = False,
    implied: bool = False,
    participant_id: Optional[str] = None,
    user_csv_path: str = "../csvs/pre_study.csv",
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Call GPT to determine the specific emotion type and associated arousal levels in the input tweet.

    Args:
        tweet_text (str): The text content of the tweet
        media_url (str, optional): URL of the image/media attached to the tweet
        include_image (bool): Whether to include image analysis in the prompt
        personalized (bool): Whether to use personalized emotion prediction (user's feelings vs general emotions)
        implied (bool): Whether to consider implied emotions not directly stated
        participant_id (str, optional): User ID to get demographic context for personalized analysis
        user_csv_path (str): Path to the CSV file containing user profiles
        debug (bool): Whether to print debug information

    Returns:
        dict: JSON response containing emotion ratings and explanation
        Note: If the implied flag is on, the response will include both explicit and implicit emotions. Remember to handle the response accordingly.
    """

    # If participant_id is provided, get user context and append to tweet text
    processed_tweet_text = tweet_text
    if participant_id:
        user_profile = get_user_profile_by_id(participant_id, user_csv_path)
        if user_profile:
            processed_tweet_text = f"{tweet_text}\n\n{user_profile}"
            personalized = (
                True  # Force personalized mode when user context is available
            )
            if debug:
                print(f"Using user context for participant {participant_id}")
        else:
            return {"error": f"Could not retrieve user profile for {participant_id}"}

    # build the schema
    if implied:
        schema = {
            "explicit": {
                "Nervous": "<1-5>",
                "Sad": "<1-5>",
                "Happy": "<1-5>",
                "Calm": "<1-5>",
                "Excited": "<1-5>",
                "Aroused": "<1-5>",
                "Angry": "<1-5>",
                "Relaxed": "<1-5>",
                "Fearful": "<1-5>",
                "Enthusiastic": "<1-5>",
                "Still": "<1-5>",
                "Satisfied": "<1-5>",
                "Bored": "<1-5>",
                "Lonely": "<1-5>",
                "explanation": "<explicit explanation>",
            },
            "implied": {
                "Nervous": "<1-5>",
                "Sad": "<1-5>",
                "Happy": "<1-5>",
                "Calm": "<1-5>",
                "Excited": "<1-5>",
                "Aroused": "<1-5>",
                "Angry": "<1-5>",
                "Relaxed": "<1-5>",
                "Fearful": "<1-5>",
                "Enthusiastic": "<1-5>",
                "Still": "<1-5>",
                "Satisfied": "<1-5>",
                "Bored": "<1-5>",
                "Lonely": "<1-5>",
                "explanation": "<implied explanation>",
            },
        }

    else:
        # Define the response schema
        schema = {
            "Nervous": "<1-5>",
            "Sad": "<1-5>",
            "Happy": "<1-5>",
            "Calm": "<1-5>",
            "Excited": "<1-5>",
            "Aroused": "<1-5>",
            "Angry": "<1-5>",
            "Relaxed": "<1-5>",
            "Fearful": "<1-5>",
            "Enthusiastic": "<1-5>",
            "Still": "<1-5>",
            "Satisfied": "<1-5>",
            "Bored": "<1-5>",
            "Lonely": "<1-5>",
            "explanation": "This tweet is <category_placeholder> because... This tweet contains <emotion_placeholder> emotion because...",
        }

    # Build the prompt based on flags
    prompt = _build_prompt(
        processed_tweet_text, include_image, personalized, implied, schema
    )
    if debug:
        print("Generated prompt:", prompt)

    # Prepare messages for API call
    messages = _prepare_messages(prompt, media_url, include_image, schema)

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=messages,
            max_tokens=500,
            temperature=0.6,
        )

        # Parse and return response
        content = response.choices[0].message.content
        if content is None:
            return {"error": "Empty response from GPT API"}

        data = json.loads(content)
        if debug:
            print("GPT response:", data)
        return data

    except Exception as e:
        if debug:
            print(f"Error calling GPT API: {e}")
        return {"error": str(e)}


def _build_prompt(
    tweet_text: str,
    include_image: bool,
    personalized: bool,
    implied: bool,
    schema: Dict[str, Any],
) -> str:
    """Build the prompt based on the configuration flags."""

    # Base emotion definitions
    emotion_definitions = """
    Definitions of emotions:

    Nervous: restless tension, emotion characterized by trembling, feelings of apprehensiveness, or other signs of anxiety or fear.

    Sad: the response to the loss of an object or person to which you are very attached. The prototypical experience is the death of a loved child, parent, or spouse. In sadness there is resignation, but it can turn into anguish in which there is agitation and protest over the loss and then return to sadness again.

    Happy: feelings that are enjoyed, that are sought by the person. A number of quite different enjoyable emotions, each triggered by a different event, involving a different signal and likely behavior. 

    Calm: free from agitation or disturbance; quiet, still, tranquil, serene. 

    Excited: a very high-intensity response to novelty and challenge, often found when there is some risk. This emotion often merges with another emotion.

    Aroused: a state of excitement or energy expenditure linked to an emotion. Usually, arousal is closely related to a person's appraisal of the significance of an event or to the physical intensity of a stimulus. Arousal can either facilitate or debilitate performance.

    Angry: the response to interference with our pursuit of a goal we care about. Anger can also be triggered by someone attempting to harm us (physically or psychologically) or someone we care about. In addition to removing the obstacle or stopping the harm, anger often involves the wish to hurt the target.

    Relaxed: abatement of intensity, vigor, energy, or tension, resulting in calmness of mind, body, or both.

    Fearful: the response to the threat of harm, physical or psychological. Fear activates impulses to freeze or flee. Often fear triggers anger.

    Enthusiastic: a feeling of excitement or passion for an activity, cause, or object.

    Still: quit, calm, motionless; stationary, remaining in the same position or attitude. 

    Satisfied: To be satisfied means to have a feeling of contentment or fulfillment.

    Bored: a state of weariness or ennui resulting from a lack of engagement with stimuli in the environment. It is often identified by individuals as a cause of feeling depressed. It can be seen as the opposite of interest and surprise.
    
    Lonely: affective and cognitive discomfort or uneasiness from being or perceiving oneself to be alone or otherwise solitary. emotional distress that results when inherent needs for intimacy and companionship are not met; unpleasant and unsettling experience that results from a perceived discrepancy (i.e., deficiency in quantity or quality) between an individual's desired and actual social relationships.
    """

    implied_expressed_definitions = ""

    # Role instruction based on personalization flag
    if personalized:
        role_instruction = """
        As an expert annotator specializing in emotions in social media content, your job is to predict what emotions and feelings the input would make a user feel when they read/view it.
        
        Consider how a typical social media user would emotionally respond to this content. Think about the emotional impact on the viewer/reader, not just the emotions expressed in the content itself.
        """

    elif implied:
        role_instruction = """
        As an expert annotator specialized in annotating emotions in social media content, your job is to analyze the explicit and implied emotions within a tweet. 
   
        Given the definitions of emotions above, evaluate if the input tweet would contain the following emotions, either explicitly or implicitly: Nervous, Sad, Happy, Calm, Excited, Aroused, Angry, Relaxed, Fearful, Enthusiastic, Still, Satisfied, Bored, Lonely.
        
        Consider any implied emotions that may not be directly stated. They might as well create an emotional impact on the user. 
        """

    else:
        role_instruction = """
        As an expert annotator specializing in emotions in social media content, your job is to predict what emotions and feelings the input would make a user feel when they read/view it.
        
        Analyze the emotional tone, sentiment, and emotional content directly expressed or implied in the tweet text.
        """

    # Content analysis instruction based on image flag
    if include_image:
        content_instruction = f"""
        Given the definitions of emotions above, evaluate if the input tweet and image would {"make the user feel" if personalized else "contain or express"} the following emotions: Nervous, Sad, Happy, Calm, Excited, Aroused, Angry, Relaxed, Fearful, Enthusiastic, Still, Satisfied, Bored, Lonely.
        
        Please address both the text and the image in your analysis.
        """
        input_description = f"""
        Input tweet: {tweet_text}
        Input image: attached in the prompt.
        """
    else:
        content_instruction = f"""
        Given the definitions of emotions above, evaluate if the input tweet would {"make the user feel" if personalized else "contain or express"} the following emotions: Nervous, Sad, Happy, Calm, Excited, Aroused, Angry, Relaxed, Fearful, Enthusiastic, Still, Satisfied, Bored, Lonely.
        """
        input_description = f"""
        Input tweet: {tweet_text}
        """

    # If implied emotions are requested, adjust the instruction
    if implied:
        implied_expressed_definitions = """
        Explicit emotions refer to emotions that are expressed through emotional words (e.g., yelled, sinking), where the words are directly associated with emotions. 
        
        Implied emotions refer to emotions that are conveyed through a series of neutral words or words that might lead to nuanced emotional impacts (e.g. sarcasm) beyond their original, individual meanings through combinatorial processing. 
        
        ——
        """

        content_instruction += """
        Consider any implied emotions that may not be directly stated but are suggested by the context or tone of the tweet. They might as well create an emotional impact on the user.
        """

    # Rating instruction
    rating_instruction = f"""
    Rate the intensity of each emotion with the following categories:
    1: Not at all — the tweet would not evoke this emotion in {"this particular user" if personalized else "general social media users"}.
    2: Slightly — the tweet would evoke this emotion slightly in {"this particular user" if personalized else "general social media users"}.
    3: Moderately — the tweet would evoke this emotion moderately in {"this particular user" if personalized else "general social media users"}.
    4: Strongly — the tweet would evoke this emotion strongly in {"this particular user" if personalized else "general social media users"}.
    5: Extremely — the tweet would evoke this emotion extremely in {"this particular user" if personalized else "general social media users"}.

    If the tweet would {"make this particular user" if personalized else "make a general social media user"} feel that emotion, assign a 2-5 to the emotion category depending on the intensity; if {"the tweet would not make the user feel such emotion" if personalized else "the emotion is not present"}, assign a 1.
    """

    # Example (adjusted based on personalization)
    if personalized:
        example = """
        Example input: I'm passionate about indie app development because I've been able to take months off at a time for my health and have no impact on my income 🩷 

        Example output: { "Nervous": 1, "Sad": 1, "Happy": 5, "Calm": 1, "Excited": 4, "Aroused": 2, "Angry": 1, "Relaxed": 1, "Fearful": 1, "Enthusiastic": 3, "Still": 1, "Satisfied": 1, "Bored": 1, "Lonely": 1, "explanation": "The tweet is likely to make a user feel extremely happy due to the use of the word 'passionate' and the heart emoji 🩷. Although the tweet conveys an enthusiastic tone, it would probably make the user feel moderately enthusiastic, as they might not be interested in app development or fully empathetic toward the author of the tweet. {give personalized reasoning}" }
        """

    elif implied:
        example = """
        Example input: The boy fell asleep and never woke up again. 

        Example output: {
        "implicit": {
            "Nervous": 3,
            "Sad": 5,
            "Happy": 0,
            "Calm": 1,
            "Excited": 0,
            "Aroused": 0,
            "Angry": 1,
            "Relaxed": 0,
            "Fearful": 4,
            "Enthusiastic": 0,
            "Still": 2,
            "Satisfied": 0,
            "Bored": 0,
            "Lonely": 2,
            "explanation": "The sentence evokes deep sadness and fear through its depiction of death, especially the peaceful yet final nature of 'falling asleep and never waking up.' Implicitly, it stirs feelings of grief, existential fear, and helplessness, particularly around themes of mortality and loss. The simplicity of the language amplifies the emotional weight."
        },
        "explicit": {
            "Nervous": 1,
            "Sad": 5,
            "Happy": 0,
            "Calm": 1,
            "Excited": 0,
            "Aroused": 0,
            "Angry": 0,
            "Relaxed": 0,
            "Fearful": 2,
            "Enthusiastic": 0,
            "Still": 3,
            "Satisfied": 0,
            "Bored": 0,
            "Lonely": 1,
            "explanation": "Explicitly, the sentence presents a peaceful but tragic event — a boy falling asleep and not waking up. The phrase directly communicates death in a calm, euphemistic way, suggesting sadness and stillness. Words like 'fell asleep' and 'never woke up' contribute to an emotionally quiet but somber tone."
        }
        }
        """

    else:
        example = """
        Example input: I'm passionate about indie app development because I've been able to take months off at a time for my health and have no impact on my income 🩷 

        Example output: { "Nervous": 1, "Sad": 1, "Happy": 4, "Calm": 2, "Excited": 3, "Aroused": 1, "Angry": 1, "Relaxed": 3, "Fearful": 1, "Enthusiastic": 5, "Still": 1, "Satisfied": 4, "Bored": 1, "Lonely": 1, "explanation": "The tweet is likely to make a user feel extremely happy due to the use of the word 'passionate' and the heart emoji 🩷. Although the tweet conveys an enthusiastic tone, it would probably make the user feel moderately enthusiastic, as they might not be interested in app development or fully empathetic toward the author of the tweet." }
        """

    # Combine all parts
    prompt = f"""
    {implied_expressed_definitions}

    {role_instruction}

    ——

    {emotion_definitions}

    ——

    {rating_instruction}

    Also, provide a brief explanation for your ratings that are not 1.{" Please address both the text and the image." if include_image else ""}

    
    ——
    {input_description}
    
    Output format: JSON object matching this structure: {json.dumps(schema, indent=2)}

    ——
    {example}
    """

    return prompt


def _prepare_messages(
    prompt: str, media_url: Optional[str], include_image: bool, schema: Dict
) -> list:
    """Prepare the messages for the OpenAI API call."""

    system_message = {
        "role": "system",
        "content": "Provide output in valid JSON with this schema:"
        + json.dumps(schema),
    }

    if include_image and media_url:
        # Include image in the user message
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": media_url}},
            ],
        }
    else:
        # Text-only message
        user_message = {"role": "user", "content": prompt}

    return [system_message, user_message]


def detect_emotion_simple(tweet_text: str) -> Dict[str, Any]:
    """
    Simplified function for basic emotion detection without images or personalization.

    Args:
        tweet_text (str): The text content of the tweet

    Returns:
        dict: JSON response containing emotion ratings and explanation
    """
    return gpt_detect_emotion(tweet_text, include_image=False, personalized=False)


def detect_emotion_personalized(tweet_text: str) -> Dict[str, Any]:
    """
    Function for personalized emotion detection (user feelings).

    Args:
        tweet_text (str): The text content of the tweet

    Returns:
        dict: JSON response containing emotion ratings and explanation
    """
    return gpt_detect_emotion(tweet_text, include_image=False, personalized=True)


def detect_emotion_with_image(
    tweet_text: str, media_url: str, personalized: bool = True
) -> Dict[str, Any]:
    """
    Function for emotion detection including image analysis.

    Args:
        tweet_text (str): The text content of the tweet
        media_url (str): URL of the image/media attached to the tweet
        personalized (bool): Whether to use personalized emotion prediction

    Returns:
        dict: JSON response containing emotion ratings and explanation
    """
    return gpt_detect_emotion(
        tweet_text, media_url=media_url, include_image=True, personalized=personalized
    )


def get_tweet_text_by_id(
    tweet_id: str, csv_path: str = "csvs/unique_tweets.csv"
) -> Optional[str]:
    """
    Retrieve tweet text by tweet ID from the unique_tweets.csv file.

    Args:
        tweet_id (str): The tweet ID to search for
        csv_path (str): Path to the CSV file containing tweets (default: "csvs/unique_tweets.csv")

    Returns:
        str or None: The tweet text if found, None if not found or error occurred
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Search for the tweet ID
        matching_rows = df[df["tweet_id"] == tweet_id]

        if len(matching_rows) > 0:
            # Return the first matching tweet text
            return matching_rows.iloc[0]["tweet"]
        else:
            print(f"Tweet ID '{tweet_id}' not found in {csv_path}")
            return None

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found")
        return None
    except KeyError as e:
        print(f"Error: Required column {e} not found in CSV file")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def detect_emotion_by_tweet_id(
    tweet_id: str,
    media_url: Optional[str] = None,
    include_image: bool = False,
    personalized: bool = False,
    implied: bool = False,
    participant_id: Optional[str] = None,
    csv_path: str = "../csvs/unique_tweets.csv",
    user_csv_path: str = "../csvs/pre_study.csv",
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Detect emotions for a tweet by its ID (retrieves text from CSV first).

    Args:
        tweet_id (str): The tweet ID to analyze
        media_url (str, optional): URL of the image/media attached to the tweet
        include_image (bool): Whether to include image analysis in the prompt
        personalized (bool): Whether to use personalized emotion prediction
        implied (bool): Whether to consider implied emotions not directly stated
        participant_id (str, optional): User ID to get demographic context for personalized analysis
        csv_path (str): Path to the CSV file containing tweets
        user_csv_path (str): Path to the CSV file containing user profiles
        debug (bool): Whether to print debug information

    Returns:
        dict: JSON response containing emotion ratings and explanation, or error message
    """
    # Get tweet text by ID
    tweet_text = get_tweet_text_by_id(tweet_id, csv_path)

    if tweet_text is None:
        return {"error": f"Could not retrieve tweet text for ID: {tweet_id}"}

    # Perform emotion detection with optional user context
    return gpt_detect_emotion(
        tweet_text,
        media_url,
        include_image,
        personalized,
        implied,
        participant_id,
        user_csv_path,
        debug,
    )


def get_user_profile_by_id(
    participant_id: str, csv_path: str = "../csvs/pre_study.csv"
) -> Optional[str]:
    """
    Retrieve user demographic profile by participant ID from the pre_study_emotion.csv file.

    Args:
        participant_id (str): The participant ID to search for
        csv_path (str): Path to the CSV file containing user profiles (default: "../csvs/pre_study.csv")

    Returns:
        str or None: Formatted string with relevant user demographics, None if not found or error occurred
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Search for the participant ID
        matching_rows = df[df["participantId"] == participant_id]

        if len(matching_rows) == 0:
            print(f"Participant ID '{participant_id}' not found in {csv_path}")
            return None

        # Get the first matching row
        user = matching_rows.iloc[0]

        # Extract relevant demographic information
        demographics = []

        # Political Party (most important for emotional impact)
        if pd.notna(user.get("Party_Gen")):
            demographics.append(f"Political Party: {user['Party_Gen']}")
        elif pd.notna(user.get("party")):
            party_mapping = {
                "rep": "Republican",
                "dem": "Democrat",
                "ind": "Independent",
            }
            party_full = party_mapping.get(user["party"], user["party"])
            demographics.append(f"Political Party: {party_full}")

        # Age
        if pd.notna(user.get("age")):
            demographics.append(f"Age: {user['age']}")

        # Gender
        if pd.notna(user.get("Gender")):
            demographics.append(f"Gender: {user['Gender']}")

        # Education
        if pd.notna(user.get("education")):
            demographics.append(f"Education: {user['education']}")

        # Race/Ethnicity
        if pd.notna(user.get("Race")):
            demographics.append(f"Race/Ethnicity: {user['Race']}")

        # Income
        if pd.notna(user.get("income")):
            demographics.append(f"Income: {user['income']}")

        # Social Status (ladder scale 1-10)
        if pd.notna(user.get("ladder")):
            ladder_value = user["ladder"]
            # Handle both numeric and string formats
            if ladder_value == "Bottom Rung 1":
                social_status = "Social Status: 1/10 (Bottom Rung - Lower Class)"
            elif ladder_value == "Top Rung 10":
                social_status = "Social Status: 10/10 (Top Rung - Upper Class)"
            else:
                # Numeric values 2-9
                try:
                    ladder_num = int(ladder_value)
                    if ladder_num <= 3:
                        status_desc = "Lower Class"
                    elif ladder_num <= 5:
                        status_desc = "Lower-Middle Class"
                    elif ladder_num <= 7:
                        status_desc = "Middle Class"
                    elif ladder_num <= 8:
                        status_desc = "Upper-Middle Class"
                    else:
                        status_desc = "Upper Class"
                    social_status = f"Social Status: {ladder_num}/10 ({status_desc})"
                except (ValueError, TypeError):
                    social_status = f"Social Status: {ladder_value}"
            demographics.append(social_status)

        # Format the output string
        if demographics:
            return "Relevant user information: " + " | ".join(demographics)
        else:
            return "Relevant user information: No demographic data available"

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found")
        return None
    except KeyError as e:
        print(f"Error: Required column {e} not found in CSV file")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def detect_emotion_with_user_context(
    tweet_text: str,
    participant_id: str,
    media_url: Optional[str] = None,
    include_image: bool = False,
    implied: bool = False,
    include_user_context: bool = True,
    csv_path: str = "../csvs/pre_study.csv",
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Detect emotions with user demographic context for personalized analysis.

    Args:
        tweet_text (str): The text content of the tweet
        participant_id (str): The participant ID for user context
        media_url (str, optional): URL of the image/media attached to the tweet
        include_image (bool): Whether to include image analysis in the prompt
        implied (bool): Whether to consider implied emotions not directly stated
        include_user_context (bool): Whether to include user demographics in the analysis
        csv_path (str): Path to the CSV file containing user profiles
        debug (bool): Whether to print debug information

    Returns:
        dict: JSON response containing emotion ratings and explanation, or error message
    """
    # Use the enhanced main function with user context
    return gpt_detect_emotion(
        tweet_text,
        media_url,
        include_image,
        personalized=True,
        implied=implied,
        participant_id=participant_id if include_user_context else None,
        user_csv_path=csv_path,
        debug=debug,
    )


def detect_emotion_by_tweet_and_user_id(
    tweet_id: str,
    participant_id: str,
    media_url: Optional[str] = None,
    include_image: bool = False,
    implied: bool = False,
    include_user_context: bool = True,
    tweet_csv_path: str = "../csvs/unique_tweets.csv",
    user_csv_path: str = "../csvs/pre_study.csv",
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Detect emotions by tweet ID and user ID with full context.

    Args:
        tweet_id (str): The tweet ID to analyze
        participant_id (str): The participant ID for user context
        media_url (str, optional): URL of the image/media attached to the tweet
        include_image (bool): Whether to include image analysis in the prompt
        implied (bool): Whether to consider implied emotions not directly stated
        include_user_context (bool): Whether to include user demographics in the analysis
        tweet_csv_path (str): Path to the CSV file containing tweets
        user_csv_path (str): Path to the CSV file containing user profiles
        debug (bool): Whether to print debug information

    Returns:
        dict: JSON response containing emotion ratings and explanation, or error message
    """
    # Use the enhanced detect_emotion_by_tweet_id function with user context
    return detect_emotion_by_tweet_id(
        tweet_id,
        media_url,
        include_image,
        personalized=True,
        implied=implied,
        participant_id=participant_id if include_user_context else None,
        csv_path=tweet_csv_path,
        user_csv_path=user_csv_path,
        debug=debug,
    )


# Example usage
if __name__ == "__main__":
    # Test the functions
    sample_tweet = (
        "Just had the most amazing coffee this morning! ☕️ Ready to tackle the day! 💪"
    )

    print("=== Simple Detection ===")
    result1 = detect_emotion_simple(sample_tweet)
    print(json.dumps(result1, indent=2))

    print("\n=== Personalized Detection ===")
    result2 = detect_emotion_personalized(sample_tweet)
    print(json.dumps(result2, indent=2))

    print("\n=== Tweet ID Lookup ===")
    # Example with a real tweet ID from the CSV
    sample_tweet_id = "tweet-1810283056890093849"
    tweet_text = get_tweet_text_by_id(sample_tweet_id)
    if tweet_text:
        print(f"Found tweet: {tweet_text[:100]}...")

        print("\n=== Emotion Detection by Tweet ID ===")
        result3 = detect_emotion_by_tweet_id(sample_tweet_id, personalized=True)
        print(json.dumps(result3, indent=2))

        print("\n=== Emotion Detection by Tweet ID with User Context ===")
        sample_participant_id = "iP009a45e51df9d4f5"
        result3b = detect_emotion_by_tweet_id(
            sample_tweet_id, participant_id=sample_participant_id
        )
        print(json.dumps(result3b, indent=2))
    else:
        print("Tweet ID not found")

    print("\n=== User Profile Lookup ===")
    # Example with a real participant ID
    sample_participant_id = "iP009a45e51df9d4f5"
    user_profile = get_user_profile_by_id(sample_participant_id)
    if user_profile:
        print(f"User profile: {user_profile}")

        print("\n=== Emotion Detection with User Context ===")
        result4 = detect_emotion_with_user_context(sample_tweet, sample_participant_id)
        print(json.dumps(result4, indent=2))

        print("\n=== Complete Analysis: Tweet ID + User ID ===")
        if tweet_text:
            result5 = detect_emotion_by_tweet_and_user_id(
                sample_tweet_id, sample_participant_id
            )
            print(json.dumps(result5, indent=2))
    else:
        print("Participant ID not found")
