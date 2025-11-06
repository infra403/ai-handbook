# 第七章：执行与编排：ToolNode总线与Handoff协议

在第六章，我们成功地让一个Agent具备了分解和规划任务的能力。这在很大程度上提升了它的自主性。然而，随着任务复杂度的进一步提升，我们会遇到一个新的瓶颈：**单一Agent的知识和能力是有限的**。

让一个Agent同时精通代码编写、数据分析、市场调研和客户沟通，就像要求一个人同时成为顶级的程序员、数据科学家、市场分析师和销售冠军一样，是不现实的。一个更合理、更可扩展的架构是构建一个**多智能体系统（Multi-Agent System）**，其中每个Agent都是一个特定领域的“专家”。

本章，我们将学习如何编排和协调一个由多个专家Agent组成的团队，让它们能够高效地分工、协作与沟通。我们将重点掌握两个核心机制：**ToolNode总线**和**Handoff协议**。

---

## 7.1 ToolNode工具总线：构建共享的“能力中心”

当系统中有多个Agent时，我们首先面临的问题是：如何管理和共享它们需要使用的工具？

一种朴素的做法是为每个Agent单独配备一套工具。但这会导致大量的重复定义和管理困难。如果一个API发生了变化，我们需要去修改所有使用到它的Agent。

一个更优雅的解决方案是构建一个**ToolNode工具总线**。

> **ToolNode工具总线** 是一种架构模式，它将系统中所有可用的工具（Functions, APIs, etc.）注册到一个统一的、中心化的节点上。任何Agent如果需要使用工具，都必须通过这个中心节点来调用。

这个模式带来了几个显而易见的好处：

-   **解耦（Decoupling）**：Agent的定义与工具的实现完全分离。Agent只需要知道“存在一个可以搜索网页的工具”，而不需要关心这个工具具体是如何实现的（是用Tavily，还是用Google）。
-   **可审计性（Auditability）**：由于所有工具调用都经过同一个“关口”，我们可以非常方便地对工具的使用情况进行集中的日志记录、权限控制和监控。
-   **动态扩展（Dynamic Extension）**：我们可以随时向总线上增加新的工具，而无需修改任何Agent的定义。Agent可以在运行时动态地“发现”这些新能力。

在`LangGraph`中，实现一个ToolNode非常简单。我们只需要创建一个专门的节点，其唯一的职责就是执行工具调用。我们在第二章中创建的`tool_node`，实际上就是一个最基础的ToolNode。

```python
# 重温我们的tool_node
from langchain_core.messages import ToolMessage

def tool_node(state: AgentState):
    """一个中心化的工具执行节点。"""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    tool_messages = []
    for call in tool_calls:
        # 根据 tool_call 的 name，从一个工具注册表中找到并执行工具
        # tool_registry 是一个简单的字典，如 {"get_weather": get_weather_func}
        tool_function = tool_registry[call["name"]]
        response = tool_function.invoke(call["args"])
        tool_messages.append(ToolMessage(content=str(response), tool_call_id=call["id"]))
    
    return {"messages": tool_messages}
```

所有Agent（无论是规划Agent还是执行Agent）在做出“调用工具”的决策后，都应该将流程导向这个**唯一的、共享的`tool_node`**。

---

## 7.2 Handoff协议：定义Agent间的“工作交接单”

有了共享的工具总线，我们解决了“能力”的共享问题。接下来是更核心的问题：多个Agent之间如何**沟通**和**交接工作**？这就是**Handoff协议（交接协议）**需要解决的问题。

> **Handoff协议** 是定义一个Agent如何将任务的控制权、上下文和必要产出，安全、清晰地传递给下一个Agent的一套“数据契约”和“流程规范”。

想象一下工厂流水线，一个工位完成了自己的工序后，需要将半成品和一张包含加工信息的“流转卡”一起放到传送带上，交给下一个工位。Handoff协议就是这张“流转卡”和“交接动作”的规范。

在`LangGraph`中，我们可以通过**扩展中心状态（State）**和**利用Pydantic模型**来优雅地实现Handoff协议。

一个典型的“规划-执行”双Agent系统的Handoff协议可以这样设计：

1.  **扩展`AgentState`**：在我们的状态对象中，增加一个专门用于任务交接的字段，例如`task_queue`。

    ```python
    from pydantic import BaseModel

    # 使用Pydantic定义一个结构化的任务
    class SubTask(BaseModel):
        agent_name: str # 指定由哪个Agent来执行
        instruction: str # 对这个子任务的具体指令
        parent_task_id: int # 标记它属于哪个父任务

    class MultiAgentState(TypedDict):
        messages: Annotated[list, lambda x, y: x + y]
        # ... 其他状态 ...
        task_queue: list[SubTask] # 一个结构化的任务队列
    ```

2.  **定义Handoff流程**：
    -   **规划Agent**在完成规划后，不直接执行。而是将分解好的子任务，构造成`SubTask`对象，放入`task_queue`中。
    -   **主路由器（Orchestrator）**会检查`task_queue`。如果队列不为空，它会取出第一个任务。
    -   主路由器根据`SubTask`中的`agent_name`，将控制流**跳转**到指定的**执行Agent**（例如`CodeExecutionAgent`或`DataAnalysisAgent`）。
    -   **执行Agent**接收到任务后，只关注`SubTask`中的`instruction`，完成自己的工作（可能会调用ToolNode），然后将结果返回。
    -   主路由器接收到执行结果，更新状态，并决定是结束，还是从队列中取出下一个任务继续执行。

这个过程清晰地定义了“谁来做”、“做什么”、“怎么交接”，使得复杂的跨Agent协作变得有序和可控。

---

## 7.3 实战Lab：构建“规划-执行”双Agent系统

让我们通过一个实战，来构建一个由“规划Agent”和“研究Agent”组成的团队。它们的任务是：**撰写一篇关于特定主题的简短研究报告。**

-   **规划Agent (PlannerAgent)**：负责将用户的研究主题分解成具体的、可执行的搜索步骤。
-   **研究Agent (ResearchAgent)**：负责接收规划Agent制定的单个步骤，调用搜索引擎执行它，并总结结果。

### 7.3.1 编写代码

创建一个新文件 `chapter_07_multi_agent.py`。

```python
import os
from typing import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# --- 1. 工具 & 模型 --- #
tools = [TavilySearchResults(max_results=3)]
# 我们需要两个模型实例，一个给规划者，一个给研究员
planner_model = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
researcher_model = ChatOpenAI(temperature=0, model_name="gpt-4-turbo").bind_tools(tools)

# --- 2. 状态 --- #
class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    # 规划Agent制定的计划
    plan: list[str]
    # 已完成的研究步骤结果
    research_results: Annotated[list, lambda x, y: x + y]

# --- 3. 节点定义 --- #

# 规划Agent节点
def planner_node(state: ResearchState):
    print("--- 规划Agent节点 ---")
    # 使用一个特定的提示来让LLM生成计划
    prompt = f"""你是一个专业的AI研究助理。根据用户的问题：'{state["messages"][-1].content}'，
    请为我制定一个不超过3个步骤的研究计划。每个步骤都应该是一个清晰、可执行的搜索查询。
    请只返回计划步骤的列表，不要有其他内容。例如：["搜索LangGraph的最新进展", "搜索ACE框架的核心思想"]"""
    
    # 调用LLM生成计划
    response = planner_model.invoke(prompt)
    plan = eval(response.content) # 注意：eval有风险，真实应用需要更安全的解析
    print(f"生成计划: {plan}")
    return {"plan": plan}

# 研究Agent节点
def researcher_node(state: ResearchState):
    print("--- 研究Agent节点 ---")
    plan = state["plan"]
    research_results = state["research_results"]
    
    # 获取当前要执行的步骤 (Handoff的核心)
    current_step_instruction = plan[len(research_results)]
    print(f"执行研究步骤: {current_step_instruction}")
    
    # 调用研究员模型来执行搜索
    response = researcher_model.invoke(current_step_instruction)
    
    # response中会包含ToolCall，这里为了简化，我们直接调用并返回文本结果
    # 真实应用中，这里应该是一个完整的ReAct子图或一个ToolNode调用
    search_result = tools[0].invoke({"query": current_step_instruction})
    
    return {"research_results": [search_result]}

# 总结节点
def summary_node(state: ResearchState):
    print("--- 总结节点 ---")
    research_results = state["research_results"]
    user_question = state["messages"][0].content
    
    summary_prompt = f"""你是一个资深的分析师。请根据以下研究结果，为用户的问题撰写一份简短的总结报告。
    
    用户问题: {user_question}
    
    研究结果:
    {research_results}
    """
    
    response = planner_model.invoke(summary_prompt)
    return {"messages": [response]}

# --- 4. 图的构建与路由 --- #

def router(state: ResearchState) -> str:
    # 如果计划还没执行完，继续研究
    if len(state.get("research_results", [])) < len(state.get("plan", [])):
        return "researcher"
    # 否则，进行总结
    return "summarizer"

graph = StateGraph(ResearchState)

graph.add_node("planner", planner_node)
graph.add_node("researcher", researcher_node)
graph.add_node("summarizer", summary_node)

graph.set_entry_point("planner")

graph.add_conditional_edges("planner", router, {"researcher": "researcher"})
graph.add_conditional_edges("researcher", router, {"researcher": "researcher", "summarizer": "summarizer"})
graph.add_edge("summarizer", END)

runnable = graph.compile()

# --- 5. 运行 --- #
if __name__ == "__main__":
    initial_input = {
        "messages": [HumanMessage(content="LangGraph和LangChain有什么区别和联系？")],
    }
    final_state = runnable.invoke(initial_input)
    print("\n--- 最终研究报告 ---")
    print(final_state["messages"][-1].content)
```

### 7.3.2 运行与解读

运行代码，你将看到一个清晰的协作流程：
1.  `planner`节点首先被调用，它将用户的模糊问题，分解成了几个具体的、可执行的搜索查询（**规划**）。
2.  控制流进入`researcher`节点，它接收到计划中的第一个步骤，并调用工具执行（**Handoff & 执行**）。
3.  `router`判断计划尚未完成，再次将流程导向`researcher`节点，执行第二个步骤。
4.  所有研究步骤完成后，`router`将流程导向`summarizer`节点。
5.  `summarizer`节点整合所有研究结果，生成最终的报告。

这个例子完美地展示了如何通过`LangGraph`的状态和条件边，实现不同角色的Agent之间的任务**编排**与**交接**。

---

## 7.4 本章小结

-   本章我们探讨了构建**多智能体系统**的必要性，即通过专家Agent团队来解决单一Agent能力不足的问题。
-   我们学习了**ToolNode工具总线**的设计模式，它通过中心化的工具管理，实现了Agent与工具的解耦、可审计性和动态扩展。
-   我们深入了**Handoff协议**的概念，并学习了如何通过扩展中心状态和使用Pydantic模型，来定义Agent之间清晰、可靠的工作交接流程。
-   通过一个“规划-执行”双Agent系统的实战Lab，我们亲手构建了一个多Agent协作的工作流，掌握了任务编排的核心技术。

到目前为止，我们的Agent不仅能独立思考，还能团队协作了。在下一章，我们将为整个系统安装一个“中枢神经”，学习如何通过**中间件（Middleware）**和**路由Agent（Router Agent）**，实现对整个Agent系统的全局治理和智能调度。
