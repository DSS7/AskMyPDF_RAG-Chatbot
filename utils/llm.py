import os
import openai

# Make sure your OpenAI API key is set as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(retrieved_chunks, query, model="gpt-3.5-turbo"):
    """
    Generates an answer to the user query based on the retrieved chunks using the new OpenAI API.
    """
    # Combine retrieved chunks into one context string
    context = "\n\n".join(retrieved_chunks)
    
    # Prepare prompt
    prompt = f"""
Use the following document excerpts to answer the question:

Document Excerpts:
{context}

Question:
{query}

Answer:
"""

    # Call OpenAI API using the new interface
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,  # lower temperature for factual answers
        max_tokens=500
    )

    # Extract generated answer
    answer = response.choices[0].message.content.strip()
    return answer
