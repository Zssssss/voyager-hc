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
        execution_error=True,
    ):
        self.ckpt_dir = ckpt_dir
        self.chat_log = chat_log
        self.execution_error = execution_error
        U.f_mkdir(f"{ckpt_dir}/action")
        if resume:
            print(f"\033[32mLoading Action Agent from {ckpt_dir}/action\033[0m")
            self.chest_memory = U.load_json(f"{ckpt_dir}/action/chest_memory.json")
        else:
            self.chest_memory = {}

        self.llm = action_model

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
        return system_message

    def render_human_message(self, *, cur_obs, code="", task="", context="", critique=""):
        cur_obs = {key:cur_obs[key] for key in ['red_obs','env_obs']}
        observation = f'Observation(The current state that can be seen, which is given in json format such as {{key:value}})：\n###\n{cur_obs}\n###\n'

        if code:
            observation += f"Code from the last round:\n{code}\n\n"
        else:
            observation += f"Code from the last round: No code in the first round\n\n"

        # if self.execution_error:
        #     if error_messages:
        #         error = "\n".join(error_messages)
        #         observation += f"Execution error:\n{error}\n\n"
        #     else:
        #         observation += f"Execution error: No error\n\n"

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

        retry = 3
        error = None
        while retry > 0:
            try:
                pattern = f"```python\n(.*?)```"
                matches = re.findall(pattern, message.content, re.DOTALL)

                assert len(matches) == 1
                matches = matches[0].strip()

                # import pdb;pdb.set_trace()
                # for match in matches:
                #     print(match)

                # 正则表达式模式，匹配以def开头，后跟任意数量的空格，然后是函数名
                func_name_pattern = r"def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\("

                # 使用findall方法查找所有匹配的函数名
                func_name_matches = re.findall(func_name_pattern, matches)
                assert len(func_name_matches) == 1
                func_name_matches = func_name_matches[0]
                

                return {
                    "program_code": matches,
                    "program_name": func_name_matches,
                    "exec_code": func_name_matches+'(obs)',
                }
            except Exception as e:
                retry -= 1
                error = e
                time.sleep(1)
        return f"Error parsing action response (before program execution): {error}"

