# 第八章：深度智能体与中间件：构建系统的“中枢神经”

随着我们系统中Agent的数量和复杂度的增加，一个新的治理问题出现了：如何应用一些**全局的、横跨所有Agent**的策略？例如，我们可能希望：

-   记录每一个进入系统的请求。
-   对所有Agent的LLM调用进行统一的成本追踪。
-   在任何危险操作（如写文件、删除数据）发生前，进行一次安全检查。
-   根据用户的身份，限制其能调用的Agent或工具。

如果将这些逻辑分散到每一个Agent或每一个节点中，将会是一场维护的噩梦。我们需要一个更高层次的抽象，一个能够拦截、处理和转发所有请求的“中枢神经系统”。本章，我们将学习两种实现这一目标的核心技术：**中间件（Middleware）**和**意图调度中心（Intent Dispatcher）**，后者通常由一个**路由Agent（Router Agent）**来承担。

---

## 8.1 中间件（Middleware）架构：为Agent执行流注入全局策略

如果你有Web开发的经验，你一定对“中间件”这个概念非常熟悉。在像Express.js或Django这样的框架中，中间件是一个函数，它在请求到达最终的处理程序（Handler）之前，或在响应返回给客户端之前，有机会对请求/响应对象进行检查和修改。

我们可以将这个强大的思想应用到我们的`LangGraph`系统中。

> **Agent中间件（Agent Middleware）** 是一种设计模式，它允许我们将一些通用的、与核心业务逻辑无关的功能（如日志、监控、认证、限流），像“管道”一样插入到Agent的执行流中。

`LangGraph`的`compile()`方法本身不直接提供一个`.use()`这样的中间件接口，但我们可以利用其`Runnable`的特性，通过**包装（Wrapping）**的方式，非常优雅地实现中间件架构。

一个`LangGraph`编译后的`Runnable`对象，其核心方法是`invoke`和`stream`。我们可以创建一个自定义的类，它也实现了这两个方法，但在调用原始`Runnable`的`invoke/stream`方法**之前**和**之后**，注入我们自己的逻辑。

### 8.1.1 `[新增]` 高级中间件模式：预算控制与人类反馈

除了基础的日志和监控，中间件还可以实现更高级、更动态的治理策略：

-   **预算控制中间件（BudgetControlMiddleware）**：这个中间件可以在内部维护一个与`thread_id`关联的成本计数器。在每次调用LLM（这通常是成本最高的部分）**之前**，它会检查累计成本是否超过了预设的阈值。如果超过，它可以采取多种策略：
    1.  **硬停止**：直接抛出异常，中断执行。
    2.  **降级**：强制后续的LLM调用使用更便宜、更快速的模型。
    3.  **请求授权**：暂停执行，并向用户或监控系统请求增加预算。

-   **人类反馈中间件（HumanFeedbackMiddleware）**：这个中间件可以在Agent执行的**每一步之后**，都暂停执行，并将当前的状态和下一步的计划展示给人类操作员，等待一个明确的“继续”指令。这在调试或高风险任务的“伴飞”模式中非常有用，它提供了一种极致的**可观测性**和**可干预性**。

### 8.1.2 实战Lab（思想）：设计一个日志中间件

让我们来设计（暂时不写完整代码，在最终项目中实现）一个可以记录每次`invoke`调用输入和输出的日志中间件。

```python
from langchain_core.runnables import Runnable

class LoggingMiddleware:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def invoke(self, input_dict, config):
        print(f"--- 中间件：收到输入 ---\n{input_dict}")
        
        # 调用原始的runnable
        result = self.runnable.invoke(input_dict, config)
        
        print(f"--- 中间件：返回输出 ---\n{result}")
        return result

    # stream方法也需要类似地包装
    def stream(self, input_dict, config):
        # ...
        yield from self.runnable.stream(input_dict, config)
        # ...

# 如何使用：
# original_runnable = graph.compile(...)
# runnable_with_logging = LoggingMiddleware(original_runnable)
# runnable_with_logging.invoke(input, config)
```

通过这种模式，我们可以创建一个中间件链，将日志、预算控制、人类反馈等多个中间件像俄罗斯套娃一样嵌套起来，构建一个极其强大的、多层次的全局治理系统。

---

## 8.2 DeepAgent意图调度中心：构建智能的“API网关”

当我们的系统拥有了多个专业的Agent（如代码Agent、研究Agent、数据分析Agent）后，我们需要一个“前台接待员”来将用户的请求，准确地分发给最合适的专家。这个“前台接待员”，就是一个**路由Agent（Router Agent）**，它构成了**意图调度中心**的核心。

> **路由Agent** 是一个特殊的、通常不执行具体业务任务的Agent。它的唯一职责是**理解（Classify）**用户的意图，并决定接下来应该调用哪个下游的Agent或工作流。

这个模式与微服务架构中的**API网关（API Gateway）**非常相似。API网关是所有外部请求的入口，它负责路由、认证、限流等，然后再将请求转发给内部的微服务。路由Agent就是我们Agent集群的“智能API网关”。

### 8.2.1 实现路由Agent

实现路由Agent的关键在于**提示工程（Prompt Engineering）**。我们需要给这个Agent一个非常明确的指令，告诉它有哪些可用的下游Agent，以及每个Agent擅长处理什么样的任务。

**代码示例：设计路由Agent的提示**

```python
# 假设我们有两个下游Agent的图 (runnable)
# code_agent_runnable: 擅长编写、修改、解释代码
# research_agent_runnable: 擅长上网搜索和总结信息

# 路由Agent的系统提示
router_prompt = """
你是一个智能的调度中心。你的任务是根据用户的问题，判断应该将任务分配给哪个专家。你必须从以下选项中选择一个：

1.  `code_agent`: 如果问题与编程、代码、算法、软件开发相关。
2.  `research_agent`: 如果问题需要最新的信息、事实核查、或对某个主题进行研究。

你必须只返回你选择的专家名字，例如 `code_agent` 或 `research_agent`。
"""

# 路由Agent的实现
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

router_model = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", router_prompt),
    ("human", "{question}")
])

# 这个chain就是一个简单的路由Agent
router_chain = prompt | router_model | StrOutputParser()
```

当用户输入“帮我用Python写一个快速排序”，`router_chain.invoke({"question": ...})` 就会返回字符串 `"code_agent"`。然后，我们的主控制流就可以根据这个字符串，来调用`code_agent_runnable`。

---

## 8.3 实战Lab：构建一个动态路由系统

让我们将上述思想结合起来，构建一个可以根据用户问题，动态选择调用“天气Agent”还是“通用聊天Agent”的系统。

创建一个新文件 `chapter_08_router_agent.py`。

```python
import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains.openai_functions import create_structured_output_runnable

# --- 1. 定义路由选择 --- #

# 使用Pydantic来定义我们希望LLM输出的结构
class RouteQuery(BaseModel):
    """根据用户问题，决定路由到哪个Agent。"""
    destination: Literal["weather_agent", "general_agent"] = "general_agent"

# --- 2. 构建路由Agent --- #

llm = ChatOpenAI(temperature=0)

# 提示工程：清晰地告诉模型它的任务和选项
system_prompt = """你是一个智能的路由分发系统。
根据用户的问题，判断应该将其路由到 `weather_agent` 还是 `general_agent`。
`weather_agent` 专门用于回答关于天气的问题。"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

# 使用特殊的runnable，它会强制LLM输出符合RouteQuery结构的JSON
router_agent = create_structured_output_runnable(RouteQuery, llm, prompt)

# --- 3. 定义下游Agent（这里用简单的函数模拟） --- #

def run_weather_agent(question: str):
    print("--- 调用天气Agent ---")
    return f"关于天气的问题 '{question}' 的答案是：晴天。"

def run_general_agent(question: str):
    print("--- 调用通用Agent ---")
    return f"关于 '{question}' 的通用答案。"

# --- 4. 运行路由系统 --- #

if __name__ == "__main__":
    questions = [
        "北京今天天气怎么样？",
        "LangGraph是什么？",
        "上海会下雨吗？"
    ]

    for q in questions:
        print(f"\n用户问题: {q}")
        # 调用路由Agent来做决策
        route = router_agent.invoke({"question": q})
        print(f"路由决策: {route.destination}")

        # 根据决策，调用相应的下游Agent
        if route.destination == "weather_agent":
            result = run_weather_agent(q)
        else:
            result = run_general_agent(q)
        
        print(f"最终结果: {result}")
```

### 8.3.1 运行与解读

运行代码，你会看到：

```
用户问题: 北京今天天气怎么样？
路由决策: weather_agent
--- 调用天气Agent ---
最终结果: 关于天气的问题 '北京今天天气怎么样？' 的答案是：晴天。

用户问题: LangGraph是什么？
路由决策: general_agent
--- 调用通用Agent ---
最终结果: 关于 'LangGraph是什么？' 的通用答案。

用户问题: 上海会下雨吗？
路由决策: weather_agent
--- 调用天气Agent ---
最终结果: 关于天气的问题 '上海会下雨吗？' 的答案是：晴天。
```

这个Lab清晰地展示了一个**路由Agent**如何作为系统的“中枢神经”，对输入进行智能分发。在`LangGraph`中，这个路由Agent本身可以是一个节点，它的输出（`weather_agent`或`general_agent`）将通过条件边，决定整个图的下一个走向，从而实现对庞大、异构的Agent集群的宏观调度。

---

## 8.4 本章小结

-   本章我们引入了**中间件（Middleware）**的思想，它是一种通过包装（Wrapping）来实现全局、跨Agent策略（如日志、监控、预算控制）的强大设计模式。
-   我们学习了**路由Agent（Router Agent）**的概念，它像一个智能API网关，负责理解用户意图，并将任务分发给最合适的下游专家Agent。
-   通过一个实战Lab，我们构建了一个可以动态选择不同Agent的**意图调度中心**，掌握了实现智能路由的核心技术。
-   我们探讨了如**预算控制**和**人类反馈**等高级中间件模式，为构建更健壮、更安全的生产级系统提供了思路。

至此，本书的第三部分“涌现代理性”已全部完成。我们已经拥有了具备高级认知能力（规划、修正）、团队协作能力（编排、交接）和全局调度能力（路由、中间件）的Agent系统。我们已经准备好迎接最终的挑战了。

在最后一部分，我们将首先讨论生产化部署和高级评估（AgentOps），然后将本书所有章节的知识融会贯通，亲手铸造我们的终极项目——`SkillCraft`框架与一个可进化的AI代码助手。
