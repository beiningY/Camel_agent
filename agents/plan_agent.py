import os
import json
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent
from retrievers import RAG
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from .text2sql_agent import Text2SQL
from colorama import Fore
class IntentResult(BaseModel):
    intent: List[str]  
    # knowledgebase_name: Optional[dict] = None
    knowledgebase_name: Optional[dict] = {"all_data": 3}
    database_query: Optional[str] = None
    
class PlanAgent:
    def __init__(self):
        """初始化PlanAgent"""
        self.load_env()
        self.load_config()
        self.init_agent()
        # 使用字典缓存不同的RAG实例，避免重复创建
        self.rag_instances = {}

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
        """初始化ChatAgent"""
        self.agent = ChatAgent(
            system_message="你是一个语义理解和意图识别方面的专家，目的是根据用户的问题进行意图识别以及需求分析。",
            model=ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                api_key=self.api_key,
                model_config_dict={"temperature": 0.4, "max_tokens": 4096},
            )
        )
    
    def get_rag_instance(self, collection_name: str) -> RAG:
        """获取RAG实例，使用缓存避免重复创建"""
        if collection_name not in self.rag_instances:
            print(f"创建新的RAG实例: {collection_name}")
            self.rag_instances[collection_name] = RAG(collection_name=collection_name)
        return self.rag_instances[collection_name]

    def plan(self, query: str) -> Optional[Dict[str, Any]]:
        """执行意图识别和需求分析"""
        if not query or not query.strip():
            return None
            
        use_msg = self._build_plan_prompt(query)            
        
        try:
            response = self.agent.step(use_msg)
            content = response.msg.content
            
            # 检查content是否为空
            if not content or not content.strip():
                print(f"警告: GPT返回空内容，使用默认配置")
                return self._get_default_plan()
                
            # 尝试解析JSON
            result_dict = json.loads(content)
            result = IntentResult(**result_dict)
            return result.dict()
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"GPT返回内容: {content}")
            print("使用默认配置继续...")
            return self._get_default_plan()
            
        except Exception as e:
            print(f"意图识别出错: {e}")
            return self._get_default_plan()
    
    def _get_default_plan(self) -> Dict[str, Any]:
        """获取默认的意图识别结果"""
        return {
            "intent": ["answer_by_knowledgebase"],
            "knowledgebase_name": {"book_zh": 3}
        }
            
    def _build_plan_prompt(self, query: str) -> str:
        """构建意图识别的提示词"""
        return f"""根据用户问题进行意图识别，返回JSON格式。

意图类型：
- answer_by_knowledgebase: 需要查询知识库
- answer_by_database: 需要查询传感器数据
- answer_by_thinking: 需要推理分析

知识库类型：
- book_zh: 养殖手册(1-5个结果)
- log: 操作日志(1-5个结果)
- all_data: 全部数据(1-5个结果)


示例输出：
{{
    "intent": ["answer_by_knowledgebase"],
    "knowledgebase_name": {{"book_zh": 3}}
}}

用户问题：{query}

请直接返回JSON，不要其他说明："""

    def data_context(self, query: str) -> str:
        """生成数据库检索提示"""
        agent = Text2SQL()
        data = agent.query_sensor_data(query)
        return data

    def knowledge_context(self, query: str, knowledgebase: Dict[str, int]) -> str:
        """知识库检索并生成提示"""
        if not knowledgebase:
            return ""
        knowledge_context = ""
        for knowledgebase_name, topk in knowledgebase.items():
            rag = self.get_rag_instance(knowledgebase_name)
            retrieved_knowledge = rag.rag_retrieve(query, topk)
            retrieved_content = knowledgebase_name + "：" + "\n".join(retrieved_knowledge)
            if retrieved_content:
                knowledge_context += retrieved_content
                
        print(Fore.YELLOW+ f"知识库检索结果:\n{knowledge_context}\n")
        return knowledge_context
    
    def cot_context(self) -> str:
        """思维链推理提示"""
        return "请一步步思考并给出答案,请进行详细的步骤分析和推理。"


    def generate_prompt(self, query: str, plan_result: Dict[str, Any]) -> str:
        """根据意图分析结果生成综合提示"""
        if not plan_result:
            return f"\n\n"
            
        prompt_parts = [f"相关信息有：\n\n"]
        intents = plan_result.get("intent", [])
        
        # 处理思维链推理
        if "answer_by_thinking" in intents:
            prompt_parts.append(self.cot_context())
        
        # 处理知识库检索
        if "knowledgebase_name" in plan_result:
            knowledgebase_names = plan_result.get("knowledgebase_name", {})
            if knowledgebase_names:
                knowledge = self.knowledge_context(query, knowledgebase_names)
                if knowledge:
                    knowledge_prompt = f"""可能的相关知识内容：
{knowledge}
请结合上述知识库内容进行分析， 如果无关则不参考。"""
                    prompt_parts.append(knowledge_prompt)
        
        # 处理数据库检索
        if "answer_by_database" in intents:
            database_query = plan_result.get("database_query")
            if database_query:
                data = self.data_context(database_query)
                if data:
                    data_prompt = f"""可能的相关的传感器数据：
{data}
请结合有关的传感器数据进行分析， 如果无关则不参考，请查询时严格注意日期时间、养殖池的编号和对应数据的匹配!!!"""
                    prompt_parts.append(data_prompt)
        
        return "\n".join(prompt_parts)

    def process_query(self, query: str) -> str:
        plan_result = self.plan(query)
        result = self.generate_prompt(query, plan_result)
        """print("="*50)
        print("意图识别和需求分析结果是："+str(plan_result))
        print("="*50)
        print("="*50)
        print("综合提示结果是："+result)"""
        return result


    