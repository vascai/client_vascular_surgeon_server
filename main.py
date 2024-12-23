from fastapi import FastAPI, HTTPException, Request
import uvicorn
from utils import *
from pinecone import Pinecone
from utils import *
from openai import OpenAI
import os
from dotenv import load_dotenv

app = FastAPI()


@app.get("/")
def hello():
  return "Hi"


@app.post("/query_ai/")
async def query_ai(request: Request):
  # Manually extract JSON from the request body
  request_data = await request.json()

  # Access the data directly from the dictionary
  query = request_data.get("query")
  namespace = request_data.get("namespace")
  previous_messages = request_data.get("previous_messages", "")
  formatted_previous_messages = format_previous_messages(previous_messages)
 
  client = OpenAI(api_key=os.environ["OPENAI_KEY"])
  pc = Pinecone(api_key=os.environ["PINECONE_KEY"])

  index_name = 'surgeon-vectordb'
  pinecone_index = pc.Index(index_name)

  query_embedding = get_embedding(client, query, model="text-embedding-3-large")
  retrieved_context = pinecone_index.query(vector=[query_embedding], top_k=25, include_metadata=True, namespace=namespace)
  context = reformat_retrieved_context(retrieved_context)

  chat_completion = client.chat.completions.create(
      model="chatgpt-4o-latest",
      messages=[
          {"role": "system", "content": get_system_prompt()},
          {"role": "user", "content": get_user_prompt(query, context, formatted_previous_messages)}
      ],
      temperature=0
  )
  answer = chat_completion.choices[0].message.content
  answer_with_sources = add_source_url(answer, context)
  print("complete")
  
  return {"answer" : answer_with_sources, "sources" : context}


if __name__ == "__main__":
  load_dotenv()
  uvicorn.run(app,host="127.0.0.1",port=int(os.getenv("PORT", 5000)))





  #all_abstracts_context = pinecone_index.query(vector=[query_embedding], top_k=25, include_metadata=True, namespace="all_abstracts")
  #trials_context = pinecone_index.query(vector=[query_embedding], top_k=3, include_metadata=True, namespace="trials")
  #guidelines_context = pinecone_index.query(vector=[query_embedding], top_k=3, include_metadata=True, namespace="guidelines")
  #context = reformat_retrieved_context(all_abstracts_context) + reformat_retrieved_context(trials_context) + reformat_retrieved_context(guidelines_context)
