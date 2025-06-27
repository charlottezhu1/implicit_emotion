#!/usr/bin/env python3
"""
Script to print the personalized prompt used for emotion detection.
This demonstrates the full prompt structure including user demographics.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.emotion_detector import get_user_profile_by_id
import json


def print_personalized_prompt(tweet_text, participant_id=None):
    """
    Print the complete personalized prompt that would be sent to GPT.
    """

    # Emotion definitions (same as in emotion_detector.py)
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

    Tired: to become weak or exhausted from exertion; to have one's strength reduced or worn out by toil or labour; to become fatigued.
    """

    # Role instruction for personalized mode
    role_instruction = """
        As an expert annotator specializing in emotions in social media content, your job is to predict what emotions and feelings the input would make a user feel when they read/view it.
        
        Consider how a typical social media user would emotionally respond to this content. Think about the emotional impact on the viewer/reader, not just the emotions expressed in the content itself.
        """

    # Content instruction for personalized mode
    content_instruction = """
        Given the definitions of emotions above, evaluate if the input tweet would make the user feel the following emotions: Nervous, Sad, Happy, Calm, Excited, Aroused, Angry, Relaxed, Fearful, Enthusiastic, Still, Satisfied, Bored, Lonely, Tired.
        """

    # Rating instruction for personalized mode
    rating_instruction = """
    Rate the intensity of each emotion with the following categories:
    1: Not at all — the tweet would not evoke this emotion in the user.
    2: Slightly — the tweet would evoke this emotion only slightly.
    3: Moderately — the tweet would evoke this emotion to a moderate degree.
    4: Strongly — the tweet would evoke this emotion strongly.
    5: Extremely — the tweet would evoke this emotion very strongly.

    If the tweet would make the user feel that emotion, assign a 2-5 to the emotion category depending on the intensity; if the tweet would not make the user feel such emotion, assign a 1.
    """

    # Example for personalized mode
    example = """
        Example input: I'm passionate about indie app development because I've been able to take months off at a time for my health and have no impact on my income 🩷 

        Example output: { "Nervous": 1, "Sad": 1, "Happy": 5, "Calm": 1, "Excited": 4, "Aroused": 2, "Angry": 1, "Relaxed": 1, "Fearful": 1, "Enthusiastic": 3, "Still": 1, "Satisfied": 1, "Bored": 1, "Lonely": 1, "Tired": 1, "explanation": "The tweet is likely to make a user feel extremely happy due to the use of the word 'passionate' and the heart emoji 🩷. Although the tweet conveys an enthusiastic tone, it would probably make the user feel moderately enthusiastic, as they might not be interested in app development or fully empathetic toward the author of the tweet." }
        """

    # Get user profile if participant_id is provided
    processed_tweet_text = tweet_text
    user_context = ""
    if participant_id:
        user_profile = get_user_profile_by_id(participant_id)
        if user_profile:
            processed_tweet_text = f"{tweet_text}\n\n{user_profile}"
            user_context = f"\n\nUSER CONTEXT ADDED:\n{user_profile}\n"
        else:
            print(f"Warning: Could not retrieve user profile for {participant_id}")

    # Input description with user context
    input_description = f"""
        Input tweet: {processed_tweet_text}
        """

    # JSON schema
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
        "Tired": "<1-5>",
        "explanation": "This tweet is <category_placeholder> because... This tweet contains <emotion_placeholder> emotion because...",
    }

    # Build complete prompt
    full_prompt = f"""{emotion_definitions}

    ——

    {role_instruction}

    {content_instruction}

    {rating_instruction}

    Also, provide a brief explanation for your ratings that are not 1.

    ——
    {input_description}
    Output format: JSON object matching this structure: {json.dumps(schema, indent=2)}

    ——
    {example}
    """

    print("=" * 80)
    print("PERSONALIZED EMOTION DETECTION PROMPT")
    print("=" * 80)

    if participant_id:
        print(f"PARTICIPANT ID: {participant_id}")
        print(user_context)
    else:
        print("PARTICIPANT ID: None (no user context)")

    print("ORIGINAL TWEET:")
    print(f"'{tweet_text}'")
    print()

    print("FULL PROMPT SENT TO GPT:")
    print("-" * 80)
    print(full_prompt)
    print("-" * 80)

    return full_prompt


if __name__ == "__main__":
    # Example usage
    sample_tweet = "The economy is doing great under this administration! Best numbers in decades! 📈"
    sample_participant_id = None  # Will be filled in during actual run

    print("Sample demonstration:")
    print_personalized_prompt(sample_tweet, sample_participant_id)

    print("\n" + "=" * 80)
    print("To use with actual user data:")
    print("python print_personalized_prompt.py")
    print("Then modify the participant_id variable in the script")
