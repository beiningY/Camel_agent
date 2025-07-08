# /usr/sarah/Camel_agent/api_server.py

from flask import Flask, request, jsonify
import logging

# 假设 ChatMultiAgent 位于 agents 子目录中
# 您需要根据实际情况调整这里的导入
try:
    from agents import ChatMultiAgent
except ImportError:
    # 处理可能的不同目录结构
    from .agents import ChatMultiAgent

# --- 初始化 ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CamelAgentAPI")

# --- 核心：在服务启动时就加载模型/智能体 ---
# 这确保了模型只被加载一次，后续请求可以复用，性能高。
logger.info("正在初始化 ChatMultiAgent 实例...")
try:
    chat_agent = ChatMultiAgent()
    logger.info("ChatMultiAgent 初始化成功！")
except Exception as e:
    logger.error(f"初始化 ChatMultiAgent 失败: {e}", exc_info=True)
    chat_agent = None


# --- API 端点 ---
@app.route('/api/run_query', methods=['POST'])
def run_query():
    if not chat_agent:
        return jsonify({"error": "专家智能体未能成功初始化，请检查服务日志。"}), 500

    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "请求体中必须包含 'query' 字段。"}), 400

    query = data['query']
    logger.info(f"收到查询请求: '{query}'")

    try:
        # 调用 run 方法
        output_msg = chat_agent.run(query)
        
        # --- 结果解析 ---
        # 根据您上次的提示，我们假设结果是一个对象或字典
        if hasattr(output_msg, 'content'):
            response_content = output_msg.content
        elif isinstance(output_msg, dict) and 'content' in output_msg:
            response_content = output_msg['content']
        else:
            response_content = str(output_msg)

        logger.info("查询成功完成。")
        return jsonify({"result": response_content})

    except Exception as e:
        logger.exception(f"执行 chat_agent.run 时出错: {e}")
        return jsonify({"error": f"执行查询时发生内部错误: {str(e)}"}), 500

# --- 启动服务 ---
if __name__ == '__main__':
    # 建议选择一个不常用的端口，避免与主应用冲突
    app.run(host='0.0.0.0', port=5001)