from dotenv import load_dotenv
import google.generativeai as genai
import json
import os
import pandas as pd

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_warmth_data(num_samples):
    generated_data = []

    warmth_definitions = """
        Warmth Scale:
        1.0: Positive. Extremely encouraging, highly decisive, very forceful, profoundly uplifting.
        0.5: Moderately encouraging, somewhat decisive, gently forceful, generally uplifting.
        0.0: Neutral, objective, factual, neither encouraging nor discouraging.
        -0.5: Slightly discouraging, somewhat wavering, mildly weak, a bit depressing.
        -1.0: Negative. Extremely discouraging, highly wavering, very weak, profoundly depressing.
    """

    few_shot_examples = [
        {
            "text": "Yoga keeps me focused. I am able to take some time for me and breath and work my body. This is important because it sets up my mood for the whole day.",
            "warmth_score": 0.6
        },
        {
            "text": "Yesterday was a very informative day.", 
            "warmth_score": 0.1
        },
        {
            "text": "I was sick with a stomach bug. I spent all day in bed and barely eating, and it was overall a pretty miserable day.",
            "warmth_score": -0.9
        },
    ]

    text_segment_to_evaluate = " There was a mountain of work so I was somewhat frustrated that many things had to be disregarded in my personal life to finish work."

    for i in range(num_samples):
        prompt_text = f"""
            You are an expert in analyzing the emotional and motivational tone of text.
            Your task is to generate a paragraph of text and then assign a "warmth" score to it.
            The warmth score should be on a continuous scale from -1.0 (extremely cool/depressing) to 1.0 (extremely warm/uplifting).

            {warmth_definitions}

            Here are some examples of text and their warmth scores:
            """
        
        for example in few_shot_examples:
            prompt_text += f"""
                <EXAMPLE>
                INPUT: {example['text']}
                OUTPUT: {{
                    "text_content": "{example['text']}",
                    "warmth_score": "{example['warmth_score']}"
                }}
                </EXAMPLE>
            """
        
        prompt_text += f"""
            Now, assign a warmth score to this new text segment:

            {text_segment_to_evaluate}

            OUTPUT:
        """

        try:
            response = model.generate_content(
                contents=prompt_text,
                generation_config=genai.types.GenerationConfig(
                    temperature=1,
                    response_mime_type='application/json',
                ),
            )

            json_output = response.text.strip()
            data_point = json.loads(json_output)
            generated_data.append(data_point)
            print(f"Generated warmth score {i+1}/{num_samples}: {data_point['warmth_score']:.2f}")
        except Exception as e:
            print(f"Error generating warmth score {i+1}: {e}")
            continue
    
    return generated_data

if __name__ == '__main__':
    generated_dataset = generate_warmth_data(num_samples=1)
    df = pd.DataFrame(generated_dataset)
    print("\nGenerated Dataframe Head:")
    print(df.head())
    df.to_csv("warmth_dataset.csv", index=False)
    print("\nDataset saved to warmth_dataset.csv")
    