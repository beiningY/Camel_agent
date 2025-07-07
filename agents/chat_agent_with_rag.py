import os
import json
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.messages import BaseMessage
from camel.societies import RolePlaying 
from retrievers import RAG

class ChatRAGAgent:
    def __init__(self):
        self.load_env()
        self.load_config()
        self.init_models()
        self.rag = RAG(self.config.get("collection_name"))
          
    def load_env(self):
        """加载环境变量"""
        load_dotenv(dotenv_path=".env")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.api_key = os.getenv("GPT_API_KEY")
        if not self.api_key:
            raise ValueError("错误：API_KEY 未在 .env 文件中或环境变量中设置。")
        os.environ["OPENAI_API_KEY"] = self.api_key

   
    def load_config(self):
        """加载配置文件"""
        try:
            with open("utils/config.json", "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print("警告: 未找到 config.json 文件，将使用默认值。")
            self.config = {}


    def init_models(self):
        """初始化模型"""
        self.model=ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                api_key=self.api_key,
                model_config_dict={"temperature": self.config.get("temperature", 0.4), "max_tokens": self.config.get("max_tokens", 4096)},
            )


    def create_society(self, query: str):
        """创建角色扮演"""
        task_kwargs = {
            'task_prompt': query if query else self.config.get("query", "你是一个南美白对虾养殖专家，请根据用户的问题，给出专业的回答。"),
            'with_task_specify': True,
            'task_specify_agent_kwargs': {
                'model': self.model,
            }
        }
        """创建用户角色"""
        user_role_kwargs = {
            'user_role_name': '南美白对虾养殖助手',
            'user_agent_kwargs': {'model': self.model}
        }
        """创建助手角色"""
        assistant_role_kwargs = {
            'assistant_role_name': '南美白对虾养殖专家',
            'assistant_agent_kwargs': {'model': self.model}
        }
        society = RolePlaying(
            **task_kwargs,          
            **user_role_kwargs,   
            **assistant_role_kwargs,   
        )
        return society
    
    def build_query_with_context(self, query: str):
        """用RAG检索并拼接上下文"""
        contexts = self.rag.rag_retrieve(query)
        context_str = "\n".join(contexts)
        return (
            f"如果提供的资料和问题有关，请根据资料回答问题，如果无关或者你没有准确的数据则说你不知道。\n\n"
            f"问题：{query}\n\n"
            f"参考资料：\n{context_str}"
        )

    def chat(self, query: str, round_limit=10):
        society = self.create_society(query)
        input_msg = society.init_chat()
        # print(f"农业专家(AI助手)系统消息:\n{society.assistant_sys_msg}\n")
        # print(f"AI用户系统消息:\n{society.user_sys_msg}\n")
        for round_idx in range(1, round_limit + 1):
            # print(f'\n===== 第{round_idx}轮 User Agent 输入 =====')
            # print(input_msg.content)
            _, user_response = society.step(input_msg)
            print(f'[AI User] {user_response.msg.content}')
            if user_response.terminated or 'CAMEL_TASK_DONE' in user_response.msg.content:
                break
            assistant_input = self.build_query_with_context(user_response.msg.content)
            # print(f'\n===== 第{round_idx}轮 Assistant Agent 输入 =====')
            # print(assistant_input)
            assistant_msg = BaseMessage.make_assistant_message(
                role_name="南美白对虾养殖专家",
                content=assistant_input
            )
            assistant_response, _ = society.step(assistant_msg)
            print(f'[AI Assistant] {assistant_response.msg.content}')
            if assistant_response.terminated:
                break
            input_msg = assistant_response.msg

if __name__ == "__main__":
    Agent = ChatRAGAgent()
    query = input("请输入问题：")
    Agent.chat(query)
