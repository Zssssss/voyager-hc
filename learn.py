from voyager import Voyager
import sys
sys.path.append('/home/ubuntu/zsss/voyager-hc/voyager/hc_env/warengine')
from voyager.llms import *
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":

    embedding_model = SentenceTransformer("/dev/pretrained_models/intfloat/e5-mistral-7b-instruct")
    embedding_model.max_seq_length = 4096
    # queries = [
    # "how much protein should a female eat",
    # "summit define",
    # ]
    # documents = [
    #     "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    #     "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
    # ]

    # query_embeddings = embedding_model.encode(queries, prompt_name="web_search_query")
    # document_embeddings = embedding_model.encode(documents)


    # client_mixtral = Client_Mixtral("/dev/pretrained_models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO") 
    client_mistral = Client_Mistral("/dev/pretrained_models/mistralai/Mistral-7B-Instruct-v0.3")

    voyager = Voyager(embedding_model=embedding_model, 
                      action_model=client_mistral, 
                      critic_model=client_mistral, 
                      curriculum_decompose_model=client_mistral,
                      curriculum_qa_model=client_mistral,
                      skill_description_model=client_mistral,
                      )
    # import pdb;pdb.set_trace()
    voyager.learn()
