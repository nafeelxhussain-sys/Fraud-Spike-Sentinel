import os
from groq import Groq

def get_fraud_explanation(txn: dict, decision: str, contribution_features: list) -> str:
    if decision == "ALLOW":
        return "Transaction normal. No anomalies detected."
        
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "System error: GROQ_API_KEY is missing."
        
    client = Groq(api_key=api_key)
        
    prompt = f"""
    You are a ruthless FinTech fraud analyst. A transaction was flagged as {decision}.
    
    Transaction Details:
    - Amount: ₹{txn.get('amount', 0)}
    - Origin Account: {txn.get('nameOrig', 'N/A')}
    
    Statistical Anomalies (Z-Scores):
    {contribution_features}
    
    Write EXACTLY ONE punchy, aggressive sentence explaining why this was blocked. 
    Focus on the feature with the highest anomaly. 
    Do not use JSON or technical ML jargon. Speak like a senior security engineer.
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq API failed: {e}"