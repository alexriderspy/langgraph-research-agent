from langchain_core.messages import AIMessage, ToolMessage

from agent.nodes import latest_hits


def test_latest_hits_takes_turns_between_tools():
    arxiv = [{"url": "a1"}, {"url": "a2"}, {"url": "a3"}]
    wiki = [{"url": "w1"}]
    hn = [{"url": "h1"}, {"url": "h2"}]
    old = ToolMessage(content="", artifact=[{"url": "old"}], tool_call_id="0")
    msgs = [old, AIMessage(content=""), *(ToolMessage(content="", artifact=h, tool_call_id=str(i))
                                          for i, h in enumerate((arxiv, wiki, hn)))]
    assert [h["url"] for h in latest_hits({"messages": msgs})] == ["a1", "w1", "h1", "a2", "h2", "a3"]
