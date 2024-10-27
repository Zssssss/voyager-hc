import re
import time
import copy
import voyager.utils as U
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate

from voyager.hc_prompts import load_prompt
from voyager.hc_control_primitives_context import load_control_primitives_context
from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage, AssistantMessage

class ActionAgent:
    def __init__(
        self,
        action_model,
        temperature=0,
        ckpt_dir="ckpt",
        resume=False,
        chat_log=True,
    ):
        self.ckpt_dir = ckpt_dir
        self.chat_log = chat_log
        U.f_mkdir(f"{ckpt_dir}/action")
        if resume:
            print(f"\033[32mLoading Action Agent from {ckpt_dir}/action\033[0m")
            self.chest_memory = U.load_json(f"{ckpt_dir}/action/chest_memory.json")
        else:
            self.chest_memory = {}

        self.llm = action_model

        self.temp_human_message_split = None
        self.temp_system_message = None


    def render_system_message(self, skills=[]):
        system_template = load_prompt("action_template")

        base_skills = [
            "move_control",
            "drop_buoy",
            "switch_infrared",
            "report_target",
            "switch_mag",
            "Dragg_control",
            "Env_Step"
        ]
        programs = "\n\n".join(load_control_primitives_context(base_skills) + skills)
        response_format = load_prompt("action_response_format")
        system_message = SystemMessage(content=system_template.format(programs=programs, response_format=response_format))
        assert isinstance(system_message, SystemMessage)
        self.temp_system_message = system_message
        return system_message

    def render_human_message(self, *, cur_obs, code="", error_messages="", task="", context="", critique=""):
        self.temp_human_message_split = {
            "cur_obs":cur_obs, "code":code, "error_messages":error_messages, "task":task, "context":context, "critique":critique
        }

        observation = ""
        if code:
            observation += f"Code from the last round:\n{code}\n\n"
        else:
            observation += f"Code from the last round: No code in the first round\n\n"

        if error_messages:
            error = "\n".join(error_messages)
            observation += f"Execution error:\n{error}\n\n"
        else:
            observation += f"Execution error: No error\n\n"

        observation += f"Chat log: None\n\n"

        cur_obs = {key:cur_obs[key] for key in ['red_obs','env_obs']}
        observation = f'Observation(The current state that can be seen, which is given in json format such as {{key:value}})：\n###\n{cur_obs}\n###\n'


        observation += f"Task: {task}\n\n"

        if context:
            observation += f"Context: {context}\n\n"
        else:
            observation += f"Context: None\n\n"

        if critique:
            observation += f"Critique: {critique}\n\n"
        else:
            observation += f"Critique: None\n\n"

        return UserMessage(content=observation)

    def process_ai_message(self, message):
        assert isinstance(message, AssistantMessage)

        def input_valid(input_txt):
            # 正则表达式匹配函数参数
            param_pattern = r'def\s+\w+\s*\((.*?)\)'

            # 搜索参数
            params_match = re.search(param_pattern, input_txt, re.DOTALL)

            # 提取参数名
            if params_match:
                params_str = params_match.group(1)  # 获取匹配的字符串
                if params_str == "Observation":
                    return True
            return False
        
        def return_valid(input_txt):
            # 正则表达式匹配 return 后跟 res
            return_pattern = r'\breturn\b.*?\bres\b'

            # 搜索 return res
            if re.search(return_pattern, input_txt, re.DOTALL):
                return True
            else:
                return False


        retry = 4
        error = None
        temp_code = message.content
        while retry > 0:
            try:
                pattern = f"```python\n(.*?)```"
                matches = re.findall(pattern, message.content, re.DOTALL)

                assert len(matches) == 1, "code blocks greater than 1 which is illegal."
                matches = matches[0].strip()

                # import pdb;pdb.set_trace()
                # for match in matches:
                #     print(match)
                temp_code = matches

                # 正则表达式模式，匹配以def开头，后跟任意数量的空格，然后是函数名
                func_name_pattern = r"def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\("

                # 使用findall方法查找所有匹配的函数名
                func_name_matches = re.findall(func_name_pattern, matches)
                assert len(func_name_matches) == 1, "multiple function generated which is illegal."
                func_name_matches = func_name_matches[0]
                

                if not input_valid(matches):
                    raise ValueError("illegal input generated which is must be Observation.")
                if not return_valid(matches):
                    raise ValueError("illegal return which must be a list of APIs.")

                return {
                    "program_code": matches,
                    "program_name": func_name_matches,
                    "exec_code": func_name_matches+'(obs)',
                }
            except Exception as e:
                retry -= 1
                import pdb;pdb.set_trace()
                self.temp_human_message_split['error_messages'] = str(e)
                self.temp_human_message_split['code'] = temp_code

                temp_human_message = self.render_human_message(**self.temp_human_message_split)
                print(f"\033[32m****Action Agent human message****\n{temp_human_message.content}\033[0m")

                message = self.llm([self.temp_system_message, temp_human_message])

                print(f"\033[34m****Action Agent ai message****\n{message.content}\033[0m")
                error = e
    
        return f"Error parsing action response (before program execution): {error}"
