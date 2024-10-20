import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaTokenizer, TextIteratorStreamer
from typing import List

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, UserMessage, SystemMessage, AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_inference.generate import generate
from mistral_inference.transformer import Transformer


class Client_Mistral:
    #https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
    def __init__(self, model_path):
        # self.tokenizer = MistralTokenizer.from_file(f"{model_path}/tokenizer.model.v3")
        # self.model = Transformer.from_folder(model_path, device="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", attn_implementation="flash_attention_2",torch_dtype=torch.float16)
        self.chatbot = pipeline("text-generation", tokenizer=self.tokenizer, model=self.model)
        
    
    def chat_completion(self, messages, max_new_tokens=2000, temperature=1.0, *args):
        # import pdb;pdb.set_trace()
        assert len(messages) == 2
        conversation = [{"role": "system", "content": messages[0].content},{"role":"user","content":messages[1].content}]

        response_data = self.chatbot(conversation, max_new_tokens=max_new_tokens, return_full_text=False)
        # import pdb;pdb.set_trace()
        return AssistantMessage(content=response_data[0]['generated_text'])
        # import pdb;pdb.set_trace()
        # completion_request = ChatCompletionRequest(messages=messages)
        # tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        # out_tokens, _ = generate([tokens], self.model, max_tokens=max_new_tokens, temperature=1.0, eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id)
        # result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        # return AssistantMessage(content=result)


    def __call__(self, messages, *args, **kwds):
        return self.chat_completion(messages=messages)
         

# class Client_Mixtral:
#     # https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO
#     def __init__(self,model_name, temperature=1.0, request_timeout=None):
#         self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name,attn_implementation="flash_attention_2", device_map="auto",torch_dtype=torch.float16)


#     def text_generation(self, messages, max_new_tokens=4000, do_sample=True, stream=True, details=True, return_full_text=False):
#         import pdb;pdb.set_trace()

#         streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, **{"skip_special_tokens": True})
#         generation_kwargs = dict(self.tokenizer(messages,return_tensors="pt"), streamer=streamer, max_new_tokens=max_new_tokens,do_sample=do_sample)
#         thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
#         thread.start()
#         thread.join()
#         while streamer.text_queue.queue[-1] is None:
#             streamer.text_queue.queue.pop()
#         temp = BaseMessage()
#         temp.content = ''.join(streamer.text_queue.queue)
#         return temp 

#     def __call__(self, messages, *args, **kwds):
#         return self.text_generation(messages=messages)


if __name__ == "__main__":
    client_mistral = Client_Mistral("/dev/pretrained_models/mistralai/Mistral-7B-Instruct-v0.3")   
    # client_mixtral = InferenceClient("/dev/pretrained_models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")
    # client_mixtral = Client_Mixtral("/dev/pretrained_models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")  