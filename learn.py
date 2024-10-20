from voyager import Voyager
import sys
sys.path.append('/home/ubuntu/zsss/voyager-hc/voyager/hc_env/warengine')
from voyager.llms import *

if __name__ == "__main__":

    
    from FlagEmbedding import FlagModel
    embedding_model = FlagModel('pretrained_models/BAAI/bge-m3',
                    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                    use_fp16=True)


    client_mistral = Client_Mistral("/dev/pretrained_models/mistralai/Mistral-7B-Instruct-v0.3")  
    # client_mixtral = Client_Mixtral("/dev/pretrained_models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO") 

    voyager = Voyager(embedding_model=embedding_model, 
                      action_model=client_mistral, 
                      critic_model=client_mistral, 
                      curriculum_decompose_model=client_mistral,
                      curriculum_qa_model=client_mistral,
                      skill_description_model=client_mistral,
                      )
    # import pdb;pdb.set_trace()
    voyager.learn()
