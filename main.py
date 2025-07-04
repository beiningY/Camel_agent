from agents import ChatMultiAgent


if __name__ == "__main__":
    chat_agent = ChatMultiAgent()
    query = "南美白对虾养殖需要注意什么？"
    chat_agent.run(query)

