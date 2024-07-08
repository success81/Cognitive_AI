from flask import Flask, request, jsonify, render_template, Response
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
import os
import time

app = Flask(__name__)

# Ensure the environment variable for Google Cloud credentials is set
if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
    raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

# Initialize Vertex AI
vertexai.init(project="clear-safeguard-420323", location="us-central1")
model = GenerativeModel("gemini-1.5-flash-001")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_response', methods=['POST'])
def generate_response():
    try:
        # Get user inputs
        cognitive_service = request.form.get("cognitive_service", "")
        challenge = request.form.get("challenge", "")
        details = request.form.get("details", "")

        if not cognitive_service or not challenge:
            raise ValueError("Cognitive service and design challenge/problem/decision are required.")

        # List of techniques based on selected cognitive service
        techniques = {
            "Brainstorming": [
                "Random Expert Ideation", "Forced Connection", "Six Thinking Hats", "Metaphorical Thinking",
                "What If", "Mash-up", "SCAMPER", "Wishing", "Concept Fan", "Lotus Blossom technique",
                "Blue sky thinking", "Analogical Transfer", "Empathy Mapping", "Scenario Thinking", "Idea Splicing",
                "Wildest Idea", "Idea expansion", "Idea inversion", "Idea transformation", "Idea randomization",
                "Idea cross-pollination", "Negative Brainstorming", "Constraint Conditions", "Random Stimuli",
                "Mindset Shifting"
            ],
            "Decision Making": [
                "SWOT Analysis", "Affinity Diagramming", "Lotus Blossom Technique", "Cost-Benefit Analysis",
                "Decision Matrix", "Force Field Analysis", "Delphi Technique", "MoSCoW Method", "Impact/Effort Matrix",
                "Kepner-Tregoe Decision Analysis", "Deconstruction", "Strategic Questioning", "Parallel Thinking",
                "Empathy Mapping", "Random Expert Ideation", "Time Travel", "Scenario Thinking", "Experience Mapping",
                "Analogical Reasoning", "Stakeholder analysis", "Paired comparison analysis", "CATWOE Analysis"
            ],
            "Problem Solving": [
                "Five Whys", "Morphological Analysis", "Problem Reversal", "Triz (Theory of Inventive Problem Solving)",
                "Dialectical Inquiry", "Forced association", "Root Cause analysis", "Fuzzy Logic", "Force Fitting",
                "Strategic Questioning", "Pareto Analysis", "Pros and Cons List", "Starbursting", "Mind Scripting",
                "Analogical Reasoning", "Scenario Planning", "Deconstruction", "Empathy Mapping", "Analogical Transfer",
                "Stakeholder analysis", "Fault Tree Analysis"
            ]
        }

        selected_techniques = techniques.get(cognitive_service, [])
        responses = []
        
        for technique in selected_techniques:
            prompt = f"""
            Do {technique} technique for the concept of {challenge}. Ensure the answer is verbose.
            Give a simple to understand definition of the {technique} technique at the beginning. Be creative.
            Here are details that can help with your response - {details}.
            """
            print(f"Sending prompt to model: {prompt}")
            response = model.generate_content(
                [prompt],
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=True,
            )
            response_text = ""
            for res in response:
                response_text += res.text
            responses.append(response_text)
            print(f"Received response for {technique}: {response_text}")
            time.sleep(8)  # Add a 5-second sleep after each LLM request

        return jsonify({"generated_text": "\n\n".join(responses)})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 2,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

