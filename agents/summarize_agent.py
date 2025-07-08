from agents.plan_agent import PlanAgent
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from dotenv import load_dotenv
import os
import json

class SummarizeAgent:
    def __init__(self):
        self.plan_agent = PlanAgent()
        self.load_env()
        self.load_config()
        self.init_agent()

    def load_env(self):
        """加载环境变量"""
        load_dotenv(dotenv_path=".env")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.api_key = os.getenv("GPT_API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY 未在 .env 文件中设置")
            
        os.environ["OPENAI_API_KEY"] = self.api_key

    def load_config(self):
        """加载配置文件"""
        with open("utils/config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)     
    
    def init_agent(self):
        self.agent = ChatAgent(
            system_message="你是一个擅长总结的专家，你的任务是根据用户的主问题，通过养殖员和专家顾问的对话形式，总结出最终的答案。",
            model=ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                api_key=self.api_key,
                model_config_dict={"temperature": 0.4, "max_tokens": 4096},
            )
        )

    def reponse_agent(self, query):
        final_query = self.plan_agent.process_query(query)
        print("="*50)
        print("最终生成的prompt:")
        print("="*50)
        print(final_query)
        response = self.agent.step(final_query)
        return response.msg.content
