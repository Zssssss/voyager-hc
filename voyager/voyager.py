import copy
import json
import os
import time
from typing import Dict

import voyager.utils as U

from .agents import ActionAgent
from .agents import CriticAgent
from .agents import CurriculumAgent
from .agents import SkillManager
from .hc_env.env_bridge import EnvBridge

class Voyager:
    def __init__(
        self,
        embedding_model,
        action_model,
        critic_model,
        curriculum_decompose_model,
        curriculum_qa_model,
        skill_description_model,
        max_iterations: int = 160,
        action_agent_temperature: float = 0,
        action_agent_task_max_retries: int = 4,
        curriculum_agent_temperature: float = 0,
        curriculum_agent_qa_temperature: float = 0,
        curriculum_agent_warm_up: Dict[str, int] = None,
        curriculum_agent_mode: str = "auto",
        critic_agent_temperature: float = 0,
        critic_agent_mode: str = "auto",
        skill_manager_temperature: float = 0,
        skill_manager_retrieval_top_k: int = 5,
        ckpt_dir: str = "ckpt",
        skill_library_dir: str = None,
        resume: bool = False,
    ):
        # init env
        self.env = EnvBridge()

        self.max_iterations = max_iterations

        # init agents
        self.action_agent = ActionAgent(
            action_model=action_model,
            temperature=action_agent_temperature,
            ckpt_dir=ckpt_dir,
            resume=resume,
        )
        self.action_agent_task_max_retries = action_agent_task_max_retries
        self.curriculum_agent = CurriculumAgent(
            embedding_model=embedding_model,
            curriculum_decompose_model=curriculum_decompose_model,
            curriculum_qa_model=curriculum_qa_model,
            temperature=curriculum_agent_temperature,
            qa_temperature=curriculum_agent_qa_temperature,
            ckpt_dir=ckpt_dir,
            resume=resume,
            mode=curriculum_agent_mode,
            warm_up=curriculum_agent_warm_up,
        )
        self.critic_agent = CriticAgent(
            critic_llm=critic_model,
            temperature=critic_agent_temperature,
            mode=critic_agent_mode,
        )
        self.skill_manager = SkillManager(
            embedding_model=embedding_model,
            skill_description_model=skill_description_model,
            temperature=skill_manager_temperature,
            retrieval_top_k=skill_manager_retrieval_top_k,
            ckpt_dir=skill_library_dir if skill_library_dir else ckpt_dir,
            resume=True if resume or skill_library_dir else False,
        )
        self.recorder = U.EventRecorder(ckpt_dir=ckpt_dir, resume=resume)
        self.resume = resume

        # init variables for rollout
        self.action_agent_rollout_num_iter = -1
        self.task = None
        self.context = ""
        self.messages = None
        # self.conversations = []
        self.last_events = None

    def reset(self, task, context=""):
        self.action_agent_rollout_num_iter = 0
        self.task = task
        self.context = context
 
        # step to peek an observation
        cur_obs = self.env.key_obs[-1]
        skills = self.skill_manager.retrieve_skills(query=self.context)
        print(f"\033[33mRender Action Agent system message with {len(skills)} skills\033[0m")
        
        
        system_message = self.action_agent.render_system_message(skills=skills)
        human_message = self.action_agent.render_human_message(cur_obs=cur_obs, code="", error_messages="", task=self.task, context=context, critique="")
        self.messages = [system_message, human_message]
        print(f"\033[32m****Action Agent human message****\n{human_message.content}\033[0m")
        assert len(self.messages) == 2
        # self.conversations = []   ---> 可以尝试用户conversation的方式生成skill， 但是考虑到prefix context太长，可能llm hold不住， 先考虑其他方法
        return self.messages

    def step(self):
        if self.action_agent_rollout_num_iter < 0:
            raise ValueError("Agent must be reset before stepping")
        
        ai_message = self.action_agent.llm(self.messages)
        print(f"\033[34m****Action Agent ai message****\n{ai_message.content}\033[0m")
        # self.conversations.append(
        #     (self.messages[0].content, self.messages[1].content, ai_message.content)
        # )  
        parsed_result = self.action_agent.process_ai_message(message=ai_message)
        import pdb;pdb.set_trace()
        success = False
        if isinstance(parsed_result, dict):
            # import pdb;pdb.set_trace()
            code = parsed_result["program_code"] + "\n" + parsed_result["exec_code"]
            try:
                
                code_return = exec(code)
                events = self.env.step(code_return)
            except Exception as e:
                events = "enviroment execute error " + str(e)
            

            # self.recorder.record(events, self.task)
            # import pdb;pdb.set_trace()
            success, critique = self.critic_agent.check_task_success(
                pre_obs=self.env.key_obs[-2],
                cur_obs=self.env.key_obs[-1],
                events=events,
                task=self.task,
                context=self.context,
                max_retries=5,
            )
            # import pdb;pdb.set_trace()
            if not success:
                # revert all the placing event in the last step
                self.env.backward()

            new_skills = self.skill_manager.retrieve_skills(query=self.context)
            system_message = self.action_agent.render_system_message(skills=new_skills)
            human_message = self.action_agent.render_human_message(
                cur_obs=self.env.key_obs[-1],
                code=parsed_result["program_code"],
                task=self.task,
                context=self.context,
                critique=critique,
            )
            self.messages = [system_message, human_message]
        else:
            assert isinstance(parsed_result, str)
            # self.recorder.record([], self.task)
            print(f"\033[34m{parsed_result} Trying again!\033[0m")
        
        # import pdb;pdb.set_trace()
        assert len(self.messages) == 2
        self.action_agent_rollout_num_iter += 1
        done = self.action_agent_rollout_num_iter >= self.action_agent_task_max_retries or success
        info = {
            "task": self.task,
            "success": success,
            "critique": '' if not success else critique,
            # "reason":'' if not success else reason,    ###加不加reason后续再看，暂且觉得一个就够了
        }
        if success:
            assert "program_code" in parsed_result and "program_name" in parsed_result, "program and program_name must be returned when success"
            info["program_code"] = parsed_result["program_code"]
            info["program_name"] = parsed_result["program_name"]
        else:
            print(f"\033[32m****Action Agent human message****\n{self.messages[-1].content}\033[0m")
        return self.messages, 0, done, info

    def rollout(self, *, task, context):
        self.reset(task=task, context=context)
        while True:
            messages, reward, done, info = self.step()
            if done:
                break
        return messages, reward, done, info

    def learn(self):
        self.env.reset()  
        obs = self.env.key_obs[-1]  ###{"red_obs": self.red_obs,"blue_obs": self.blue_obs, "env_obs": self.env_obs, "obs_message": obs_message} 
        last_task_critique, lask_task = "", ""      ###其实可以用对话的方式改进的，但是怕支持的上下文不够，先这样

        while True:
            if self.recorder.iteration > self.max_iterations:
                print("Iteration limit reached")
                import pdb;pdb.set_trace()
                ###卡壳了，手动干预
                break
            # import pdb;pdb.set_trace()
            task, context = self.curriculum_agent.propose_next_task(obs=obs, lask_task_critique=last_task_critique, last_task=lask_task)
            print(
                f"\033[35mStarting task {task} for at most {self.action_agent_task_max_retries} times\033[0m"
            )
            try:
                messages, reward, done, info = self.rollout(task=task,context=context)
            except Exception as e:
                import pdb;pdb.set_trace()
                ###报到这肯定有错了， for debug
                info = {"task": task,"success": False}
                # reset bot status here
                self.env.backward()
                # use red color background to print the error
                print("Your last round rollout terminated due to error:")
                print(f"\033[41m{e}\033[0m")

            if info["success"]:
                self.skill_manager.add_new_skill(info)
            else:
                #### reconstruct next task 
                obs = self.env.key_obs[-1]
                self.recorder.iteration += 1   ####recorder/resume的那个类还没改好，todo
                last_task_critique = info['critique']
                last_task = task

            # self.curriculum_agent.update_exploration_progress(info)
            print(f"\033[35mCompleted tasks: {', '.join(self.curriculum_agent.completed_tasks)}\033[0m")
            print(f"\033[35mFailed tasks: {', '.join(self.curriculum_agent.failed_tasks)}\033[0m")

        return {
            "completed_tasks": self.curriculum_agent.completed_tasks,
            "failed_tasks": self.curriculum_agent.failed_tasks,
            "skills": self.skill_manager.skills,   ###最后返回的生成的技能树组成的库， 外部只是图谱可以直接加进去，虚拟环境下可以自动探索生成些
        }

    def decompose_task(self, task):   ###分解任务的函数，针对比较困难的任务做改进，未改， todo
        if not self.last_events:
            self.last_events = self.env.reset()
        return self.curriculum_agent.decompose_task(task, self.last_events)

    
    def learn_after_plan(self):  ####改learn函数过程，先进行思维链规划，再逐个进行sub_task的探索，TODO
        pass
    
    
    def inference(self, task=None, sub_goals=[], reset_mode="hard", reset_env=True):     ###用现有技能库进行虚拟对局的函数，未改， todo
        if not task and not sub_goals:
            raise ValueError("Either task or sub_goals must be provided")
        if not sub_goals:
            sub_goals = self.decompose_task(task)
        self.env.reset()
        self.curriculum_agent.completed_tasks = []
        self.curriculum_agent.failed_tasks = []
        self.last_events = self.env.step("")
        while self.curriculum_agent.progress < len(sub_goals):
            next_task = sub_goals[self.curriculum_agent.progress]
            context = self.curriculum_agent.get_task_context(next_task)
            print(
                f"\033[35mStarting task {next_task} for at most {self.action_agent_task_max_retries} times\033[0m"
            )
            messages, reward, done, info = self.rollout(
                task=next_task,
                context=context,
                reset_env=reset_env,
            )
            self.curriculum_agent.update_exploration_progress(info)
            print(
                f"\033[35mCompleted tasks: {', '.join(self.curriculum_agent.completed_tasks)}\033[0m"
            )
            print(
                f"\033[35mFailed tasks: {', '.join(self.curriculum_agent.failed_tasks)}\033[0m"
            )

