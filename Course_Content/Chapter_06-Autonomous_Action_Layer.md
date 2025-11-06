# 第三部分：涌现代理性——在控制下释放自主能力

---

# 第六章：自主行动层：融合ReAct与ACE思想的执行引擎

在本书的前半部分，我们专注于为Agent构建一个坚固的“底盘”——一个可观测、可干预、可记忆、可获取外部知识的**可控性框架**。现在，是时候在这个底盘之上，安装一个强大的“引擎”了。这个引擎，就是Agent的**自主行动层**。

本章，我们将深入探索**代理性（Agenticness）**的实现。我们将学习如何让Agent从一个简单的“工具调用者”，进化为一个能够面对复杂目标、自主进行**任务规划**、**多步执行**乃至**自我修正**的智能实体。我们将从`ReAct`模式的工程化实现入手，并从`ACE`、`Manus`和`DeepAgents`等前沿研究中汲取核心思想。

---

## 6.1 代理性的基石：ReAct思维链与执行链

我们在第一章提到过**ReAct (Reason + Act)** 循环。它是几乎所有现代Agent的理论基石。现在，我们需要从工程实现的角度来重新审视它。

一个简单的ReAct循环，在代码层面通常表现为一个`while`循环，这存在一些问题：

-   **状态管理混乱**：循环中的所有变量（如思考过程、行动历史）都散落在局部变量中，难以追踪和持久化。
-   **可控性差**：很难在循环的特定步骤（如“思考”之后，“行动”之前）注入干预逻辑。
-   **难以扩展**：如果想加入更复杂的逻辑，如“规划”或“自我修正”，`while`循环会变得极其臃肿和复杂。

`LangGraph`为我们提供了一个完美的解决方案：**将隐式的ReAct循环，显式地建模为一个状态图**。在这个图中，我们可以将“思考”和“行动”清晰地分离为不同的节点。

-   **`agent` 节点 (思考链)**：这个节点专门负责“思考”。它接收当前的所有信息（用户问题、历史记录、工具执行结果），然后决定下一步应该做什么。它的输出应该是对下一步行动的决策，例如一个`ToolCall`。
-   **`tool_node` 节点 (行动链)**：这个节点专门负责“行动”。它接收`agent`节点的决策，并忠实地执行它（调用工具）。它的输出是工具执行的结果。

我们在第二章构建的“天气Agent”正是这个模式的最简实现。通过将思考和行动分离到不同的节点，我们获得了关键的**可观测性**和**可干预性**：我们可以在两个节点之间插入逻辑，例如在工具执行前进行审批，或者在工具执行后进行结果校验。

---

## 6.2 从框架中汲取智慧：ACE、Manus与DeepAgents的核心思想

简单的ReAct循环足以应对一步或两步就能完成的简单任务。但面对“写一个功能，并为它创建单元测试”这样的复杂任务时，Agent就需要更高级的认知能力。学术界和工业界对此进行了大量探索，我们可以从中汲取宝贵的思想。

### 6.2.1 ACE (Agent-Cycle-Environment)：明确Agent与环境的交互

`ACE`框架强调将Agent的执行流程，明确地定义为一个与**环境（Environment）**交互的主循环。这个思想非常契合我们的`LangGraph`模型。

-   **Agent**：就是我们的`agent`节点，负责决策。
-   **Cycle**：就是我们的`LangGraph`图本身，它定义了Agent如何从一个状态流转到下一个状态。
-   **Environment**：就是我们的工具集（如文件系统、数据库、API）。`tool_node`是Agent与环境交互的唯一接口。

采用`ACE`思想来设计我们的图，意味着我们需要让Agent的每一步行动都通过与“环境”的交互来完成，并从环境中获得“观察结果”。例如，Agent不应该“在脑子里”想象代码写好了，它必须**行动**（调用`write_file`工具），然后**观察**（调用`read_file`或`run_tests`工具来查看结果），再进行下一步**思考**。

### 6.2.2 Manus & DeepAgents：“规划-执行-自校正”的闭环

`Manus`和`DeepAgents`等更前沿的框架，引入了比`ReAct`更宏大的认知循环，其核心思想可以概括为 **“规划-执行-自校正”**。

-   **规划（Plan）**：在开始任何具体行动之前，Agent首先对整个任务进行思考，并将其分解成一个包含多个步骤的**计划**。例如，对于“添加用户登录功能”，计划可能是：`[1. 创建user_controller.py, 2. 在其中添加login函数, 3. 创建对应的测试文件, 4. 编写测试用例, 5. 运行测试]`。

-   **执行（Execute）**：Agent按照计划，一步一步地执行。每一步都是一个简单的`ReAct`循环（思考->调用工具）。

-   **自校正（Self-Correct）**：在执行过程中，如果某一步的结果与预期不符（例如，测试运行失败），Agent不会立即放弃。它会进入“自校正”模式，将**原始目标、失败的计划步骤、以及错误信息**一起作为新的输入，重新进行思考，生成一个**修正计划**（例如，“读取刚刚失败的测试的报错信息，并修改`user_controller.py`中的代码以修复bug”），然后继续执行。

### 6.2.3 互补关系：从微观到宏观

> **ReAct是微观层面的“一步”，而ACE和Manus是宏观层面的“完整任务循环”。**

-   我们的Agent在执行计划中的**每一步**时，遵循的是`ReAct`模式（思考调用哪个工具，然后执行）。
-   整个任务的完成过程，从接收任务到最终交付，遵循的是`ACE`或`Manus`的宏观循环（规划->执行->观察->修正...）。

`LangGraph`的灵活性让我们可以在同一个图中，同时实现这两种层级的循环，构建出真正强大的自主智能体。

---

## 6.3 实战Lab：构建一个具备初步规划能力的Agent

让我们来升级第二章的Agent，让它能够处理一个需要“多步思考”的任务。我们将不直接实现完整的“自校正”循环（这将在最终项目中完成），而是先实现一个能**生成计划**并**按步骤执行**的Agent。

**任务**：用户提出一个问题，Agent需要先上网搜索相关信息，然后基于搜索结果进行总结和回答。

这需要两个步骤，因此Agent必须先“规划”。

### 6.3.1 环境与工具准备

1.  **安装依赖**：我们需要一个用于Web搜索的工具。`Tavily`是一个专为LLM Agent设计的优秀搜索引擎。

    ```dockerfile
    # .devcontainer/Dockerfile
    RUN pip install --no-cache-dir \
        # ... (原有依赖)
        tavily-python
    ```
    同时，在`.env`文件中添加你的Tavily API密钥。
    ```
    # .env
    TAVILY_API_KEY="your_tavily_api_key_here"
    ```
    重建你的Dev Container。

2.  **定义工具**：

    ```python
    from langchain_community.tools.tavily_search import TavilySearchResults

    # Tavily搜索工具
    tavily_tool = TavilySearchResults(max_results=3)
    tools = [tavily_tool]
    ```

### 6.3.2 构建“规划-执行”图

我们将设计一个新的图，它包含一个专门的`planner`节点。

创建一个新文件 `chapter_06_planner_agent.py`。

```python
import os
from typing import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# --- 1. 定义工具和模型 ---
tools = [TavilySearchResults(max_results=3)]
model = ChatOpenAI(temperature=0).bind_tools(tools)

# --- 2. 定义状态 --- #
# 我们增加 plan 和 past_steps 字段来追踪计划和执行历史
class PlannerState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    plan: list[str]
    past_steps: Annotated[list, lambda x, y: x + y]

# --- 3. 定义节点 --- #

# 规划器节点：接收用户问题，生成一个计划
def planner_node(state: PlannerState):
    print("--- 规划器节点 ---")
    messages = state["messages"]
    # 创建一个提示，要求LLM生成计划
    planner_prompt = f"""你是一个聪明的AI助手。根据用户的问题：'{messages[-1].content}'，
    你需要制定一个由以下两个步骤组成的计划来回答它：
    1. 使用搜索工具查找相关信息。
    2. 总结搜索到的信息并回答用户的问题。
    请严格按照这个格式输出你的计划。"""
    
    # 这里为了简化，我们直接硬编码了计划。在真实应用中，应该由LLM生成。
    plan = [
        "1. 使用搜索工具查找相关信息。",
        "2. 总结搜索到的信息并回答用户的问题。"
    ]
    print(f"生成计划: {plan}")
    return {"plan": plan}

# 执行节点：执行计划中的一步
def execute_step_node(state: PlannerState):
    print("--- 执行步骤节点 ---")
    plan = state["plan"]
    past_steps = state["past_steps"]
    
    # 获取当前要执行的步骤
    current_step = plan[len(past_steps)]
    print(f"执行步骤: {current_step}")
    
    # 使用LLM来执行这一步
    # 我们将整个状态（包括计划和历史）都提供给它
    response = model.invoke(state["messages"] + [HumanMessage(content=f"我正在执行计划的这一步: {current_step}。请执行它。")])
    
    # 返回LLM的响应，并更新已完成步骤
    return {
        "messages": [response],
        "past_steps": past_steps + [current_step]
    }

# --- 4. 定义图和边 --- #

def router(state: PlannerState) -> str:
    print("--- 路由器 ---")
    # 如果计划还没执行完，继续执行下一步
    if len(state["past_steps"]) < len(state["plan"]):
        print("决策: 继续执行计划")
        return "execute_step"
    else:
        print("决策: 计划执行完毕，结束")
        return "__end__"

graph = StateGraph(PlannerState)
graph.add_node("planner", planner_node)
graph.add_node("execute_step", execute_step_node)

graph.set_entry_point("planner")

graph.add_edge("planner", "execute_step")
graph.add_conditional_edges("execute_step", router, {"execute_step": "execute_step", "__end__": END})

runnable = graph.compile()

# --- 5. 运行 --- #
if __name__ == "__main__":
    initial_input = {
        "messages": [HumanMessage(content="LangGraph是什么？它和LangChain有什么区别？")],
        "past_steps": []
    }
    for step in runnable.stream(initial_input):
        print(step)
        print("---")
```

### 6.3.3 运行与解读

运行代码，你会看到Agent首先进入`planner_node`制定计划，然后两次进入`execute_step_node`。第一次，它会调用搜索工具；第二次，它会基于第一次的搜索结果进行总结，最终完成任务。这个过程清晰地展示了Agent是如何将一个复杂任务分解为多个简单步骤并依次执行的。

这个例子虽然简单（计划是硬编码的），但它为你揭示了通往高级Agent的大门。在本书的最终项目中，我们将把这个模式发扬光大，实现一个能由LLM**动态生成计划**、并在执行失败时**动态生成修正计划**的、真正强大的`OrchestratorAgent`。

---

## 6.4 本章小结

-   本章我们从工程角度重新审视了**ReAct**模式，并学习了如何使用`LangGraph`将其从隐式的`while`循环，重构为显式的、可控的“思考-行动”状态图。
-   我们从`ACE`、`Manus`和`DeepAgents`等前沿研究中汲取了核心思想，理解了高级Agent所需具备的**“规划-执行-自校正”**的宏观认知循环。
-   我们明确了`ReAct`与`ACE/Manus`之间的互补关系：前者是执行计划中每一步的微观循环，后者是完成整个复杂任务的宏观框架。
-   通过一个实战Lab，我们构建了一个具备初步**规划能力**的Agent，它能够将一个两步任务分解并按顺序执行，为我们最终实现更高级的自主智能体打下了基础。

在下一章，我们将探讨当系统中存在多个Agent时，它们应如何有效地沟通和协作。我们将学习**ToolNode总线**和**Handoff协议**，进入多智能体编排的世界。
