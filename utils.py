


def get_embedding(client, text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


def get_system_prompt():
    return ( 
        "You are an expert vascular surgeon assisting surgeons with their practice."
        "You have access to the most up-to-date textbooks, guidelines, and clinical trials through the Source. "
        "Determine the best answer to the QUESTION using Sources provided. Provide a detailed yet succinct explanation to another physician. "
        "Cite a footnote next to each source you use in the format [Source i] and use as many sources as necessary.\n"
    )

def format_previous_messages(messages):

  if messages == "":
    return ""

  formatted_messages = ""
  for i in range(len(messages)):
    current_user_message = messages[i]["user_query"] 
    current_ai_message = messages[i]["ai_response"] 

    formatted_messages += f"User Question: {current_user_message}\nAI Response: {current_ai_message}\n\n"

  return formatted_messages
    

def get_user_prompt(question, context, formatted_previous_messages):

    sources = ""

    for i, context in enumerate(context):
        sources += f"Source {i+1}\n{context['text']}\n\n"


    final_template = \
f"""
SOURCES:
{sources}
PREVIOUS CONVERSATIONS:
{formatted_previous_messages}
QUESTION:
{question}
"""  

    return final_template


def add_source_url(answer, context):

    source_added = False 

    for i in range(0, len(context)):
        source_placeholder_v1 = f"[Source {i + 1}]"
        source_placeholder_v2 = f"(Source {i + 1})"
        url = context[i]['original_link']
        html_link = ""
        if source_added:
            html_link = f'<sup><a href="{url}" style="text-decoration: none;">, {i + 1}</a></sup>'
        else:
            html_link = f'<sup><a href="{url}" style="text-decoration: none;">{i + 1}</a></sup>'


        if source_placeholder_v1 in answer or source_placeholder_v2 in answer:
            source_added = True
        answer = answer.replace(source_placeholder_v1, html_link)
        answer = answer.replace(source_placeholder_v2, html_link)

    return answer

def reformat_retrieved_context(context):
    context_arr = context['matches']

    res = []

    for context in context_arr:
        metadata = context['metadata']
        res.append({'title' : metadata['title'], 'text' : metadata['text'], 'original_link' : metadata['original_link'],})
    return res