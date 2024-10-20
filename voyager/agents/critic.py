from voyager.hc_prompts import load_prompt
from voyager.utils.json_utils import fix_and_parse_json
import copy
from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage, AssistantMessage

class CriticAgent:
    def __init__(
        self,
        critic_llm,
        temperature=0,
        mode="auto",
    ):
        # self.llm = Client_Mistral(
        #     model_name="/dev/pretrained_models/mistralai/Mistral-7B-Instruct-v0.3",
        #     temperature=temperature,
        #     request_timeout=request_timout,
        # )
        self.llm = critic_llm
        assert mode in ["auto", "manual"]
        self.mode = mode

    def render_system_message(self):
        system_message = SystemMessage(content=load_prompt("critic"))
        return system_message

    def render_human_message(self, *, pre_obs, cur_obs, events, task, context):
        pre_obs = {key:pre_obs[key] for key in ['red_obs','env_obs']}
        cur_obs = {key:cur_obs[key] for key in ['red_obs','env_obs']}
        observation = f'Pre_observation(The state before the action is performed that can be seen, which is given in json format such as {{key:value}})：\n###\n{pre_obs}\n###\n'
        
        observation += f'Cur_observation(The current state that can be seen, which is given in json format such as {{key:value}})：\n###\n{cur_obs}\n###\n'

        observation += f'Action_event: {events}\n'

        observation += f"Task: {task}\n\n"

        if context:
            observation += f"Context: {context}\n\n"
        else:
            observation += f"Context: None\n\n"

        print(f"\033[31m****Critic Agent human message****\n{observation}\033[0m")
        return UserMessage(content=observation)

    def human_check_task_success(self):
        confirmed = False
        success = False
        critique = ""
        while not confirmed:
            success = input("Success? (y/n)")
            success = success.lower() == "y"
            critique = input("Enter your critique:")
            print(f"Success: {success}\nCritique: {critique}")
            confirmed = input("Confirm? (y/n)") in ["y", ""]
        return success, critique

    def ai_check_task_success(self, messages, max_retries=5):
        if max_retries == 0:
            print(
                "\033[31mFailed to parse Critic Agent response. Consider updating your prompt.\033[0m"
            )
            return False, ""

        if messages[1] is None:
            return False, ""

        critic = self.llm(messages).content
        print(f"\033[31m****Critic Agent ai message****\n{critic}\033[0m")
        import pdb;pdb.set_trace()
        try:
            response = fix_and_parse_json(critic)
            assert response["success"] in [True, False]
            if "critique" not in response:
                response["critique"] = ""
            return response["success"], response["critique"]
        except Exception as e:
            print(f"\033[31mError parsing critic response: {e} Trying again!\033[0m")
            return self.ai_check_task_success(
                messages=messages,
                max_retries=max_retries - 1,
            )

    def check_task_success(self, *, pre_obs, cur_obs, events, task, context, max_retries=5):
        human_message = self.render_human_message(
            pre_obs=pre_obs,
            cur_obs=cur_obs,
            events=events,
            task=task,
            context=context,
        )

        messages = [
            self.render_system_message(),
            human_message,
        ]

        if self.mode == "manual":
            return self.human_check_task_success()
        elif self.mode == "auto":
            return self.ai_check_task_success(messages=messages, max_retries=max_retries)
        else:
            raise ValueError(f"Invalid critic agent mode: {self.mode}")
