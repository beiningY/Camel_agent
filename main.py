from agents import ChatMultiAgent
from retrievers import ModelManager

def preload_models():
    """预热模型"""
    print("正在预热模型...")
    model_manager = ModelManager()
    # 预加载embedding模型
    model_manager.get_embedding_model()
    print("模型预热完成！")

if __name__ == "__main__":
    # 加载模型
    preload_models()
    # 创建多智能体聊天系统
    chat_agent = ChatMultiAgent()
    # 可以尝试多个查询，观察模型不会重复加载
    queries = [
        "南美白对虾养殖需要注意什么？",
        "循环水养殖系统的优势有哪些？",
        "如何监控水质参数？"
    ]
    
    print("\n" + "="*50)
    print("开始多轮对话测试...")
    print("="*50)
    
    for i, query in enumerate(queries, 1):
        print(f"\n第{i}轮查询: {query}")
        print("-" * 30)
        chat_agent.run(query)
        print("-" * 30)
        if i < len(queries):
            input("\n按Enter键继续下一轮...")

