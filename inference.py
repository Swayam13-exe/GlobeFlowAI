import os
import json
from openai import OpenAI
from env.environment import GlobeFlowEnv

def run_agent():
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        hf_token = os.environ.get("OPENAI_API_KEY", "dummy")

    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token
    )

    # Use 'hard' to show conflict resolution capabilities, can be easy/medium
    env = GlobeFlowEnv(task_level="hard")
    state = env.reset()

    print("--- Starting Inference ---")

    for step in range(10):
        print(f"\\nStep {step+1}")
        
        prompt = f"""
The agent must:
1. Analyze state
2. Identify missing requirements
3. Choose ONE best action

Decision priority:
1. Missing documents → request
2. Submitted docs → verify
3. Approvals → follow HR → Legal → Finance
4. Compliance → set payroll/tax
5. Finish → finalize_case

Rules:
* Do not repeat actions
* Do not violate dependencies
* Do not finalize early

Output format STRICT:
{{
"action_type": "...",
"target": "..."
}}

No explanation. Only JSON.

Current State:
{json.dumps(state, indent=2)}
"""

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            # Clean up markdown if any
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "").strip()
            if "```" in content:
                content = content.replace("```", "").strip()
                
            action = json.loads(content)
            print(f"Action: {action}")
            
        except Exception as e:
            print(f"Failed to parse action or API Error: {e}")
            break
            
        next_state, reward, done, info = env.step(action)
        print(f"Reward: {reward}")
        
        state = next_state
        if done:
            print(f"Finished. Final Score: {info.get('score', 0.0)}")
            break

if __name__ == "__main__":
    run_agent()
