# 第二章：技术栈概览与“Hello, Controllable World!”

在第一章，我们建立了本书的核心思想：在**代理性**与**可控性**之间取得平衡。从本章开始，我们将把理论付诸实践。本章的目标是搭建一个功能完备、可复现的开发环境，并构建我们的第一个、最简单的智能体，亲眼见证它的工作方式。

---

## 2.1 环境搭建：使用Docker和VSCode Dev Containers

在数据密集型应用或任何复杂的软件工程项目中，保证开发环境的一致性是成功的先决条件。我们不希望因为Python版本、库依赖或操作系统差异等问题浪费时间。因此，我们将采用目前业界最流行的解决方案：**Docker + VSCode Dev Containers**。

-   **Docker**：一个开源的应用容器引擎，可以将我们的应用及其所有依赖（库、运行时等）打包到一个轻量级、可移植的容器中。
-   **VSCode Dev Containers**：一个VSCode扩展，它允许我们直接在一个Docker容器内部进行开发、调试和运行代码，就像在本地环境中一样流畅，同时享受容器化带来的隔离性和一致性。

### 2.1.1 准备工作

在开始之前，请确保您的电脑上已经安装了以下软件：

1.  [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2.  [Visual Studio Code](https://code.visualstudio.com/)
3.  在VSCode中安装 [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 扩展。

### 2.1.2 配置开发容器

现在，让我们为我们的项目创建一个专业的开发容器。请在您的项目根目录下（例如 `/Users/xp/Documents/airport/saas/agent_course`）创建一个名为 `.devcontainer` 的新目录，并在其中创建两个文件。

**1. `.devcontainer/devcontainer.json`**

这个文件告诉VSCode如何构建和配置我们的开发容器。

```json
{
	"name": "AI Agent Course",
	"build": {
		"dockerfile": "Dockerfile"
	},
	"customizations": {
		"vscode": {
			"extensions": [
				"ms-python.python",
				"ms-python.vscode-pylance"
			]
		}
	},
    "runArgs": [
        "--env-file",
        ".env"
    ]
}
```

-   `name`: 为你的开发容器起一个名字。
-   `build.dockerfile`: 指定用于构建容器镜像的Dockerfile文件。
-   `customizations.vscode.extensions`: 指定容器启动后，需要自动在VSCode中安装的扩展列表。这里我们安装了官方的Python支持和Pylance语言服务器。
-   `runArgs`: 这是一个非常重要的配置。它告诉Dev Containers在启动容器时，读取项目根目录下的`.env`文件，并将其中的内容作为环境变量注入到容器中。这将是我们存放API密钥等敏感信息的最佳实践。

**2. `.devcontainer/Dockerfile`**

这个文件定义了如何构建我们的Docker镜像，包括基础系统、Python版本和所需的库。

```dockerfile
# 使用官方的Python 3.11镜像作为基础
FROM python:3.11-slim

# 设置工作目录
WORKDIR /workspace

# 安装核心依赖
RUN pip install --no-cache-dir \
    langchain \
    langgraph \
    pydantic \
    "langchain_openai[openai]" \
    python-dotenv

# 将项目文件复制到工作区
COPY . /workspace

# 安装项目特定的依赖（如果requirements.txt存在）
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
```

**3. `.env` 文件**

在您的项目根目录下（与`.devcontainer`同级）创建一个名为`.env`的文件。**切记，不要将此文件提交到Git仓库中！**

```
# .env
# 将此处替换为你的DeepSeek API密钥
# 你可以在 https://platform.deepseek.com/api_keys 获取
DEEPSEEK_API_KEY="your_deepseek_api_key_here"

# DeepSeek与OpenAI的API格式兼容，所以我们可以这样设置
OPENAI_API_KEY="your_deepseek_api_key_here"
OPENAI_API_BASE="https://api.deepseek.com/v1"
```

### 2.1.3 启动开发容器

现在，所有配置都已完成。在VSCode中，按下 `F1` 打开命令面板，输入并选择 `Dev Containers: Reopen in Container`。VSCode会自动开始构建Docker镜像并在容器中重新打开您的项目。第一次启动会花费几分钟时间，之后会非常快。

当左下角的状态栏显示 `Dev Container: AI Agent Course` 时，恭喜你，你已经进入了一个完全配置好的、隔离的开发环境！

---

## 2.2 关键框架的哲学

在我们的课程中，我们不会堆砌大量的框架，而是精选了三个核心的、设计哲学互补的库。理解它们各自的定位至关重要。

-   **`LangChain`：Agent开发的“基础类库”**
    你可以将LangChain想象成Java世界中的`Apache Commons`或Python标准库中的`collections`。它为我们提供了构建LLM应用所需的大量、高质量的“积木”。例如：
    -   **模型I/O**：提供了统一的接口来与各种LLM（如DeepSeek, OpenAI, Anthropic）进行交互。
    -   **提示模板（Prompt Templates）**：帮助我们动态地、安全地构建复杂的提示。
    -   **输出解析器（Output Parsers）**：将LLM返回的纯文本，解析成我们需要的结构化数据（如JSON、Pydantic对象）。
    **一句话定位**：LangChain是我们的“瑞士军刀”，提供了构建Agent所需的、可重用的基础组件。

-   **`LangGraph`：实现“可控性”的“状态机引擎”**
    如果说LangChain是积木，那么LangGraph就是将这些积木搭建成一个**可控、可观测的复杂结构**的“图纸”和“脚手架”。它将Agent的执行流程从一个简单的、不可控的`while`循环，升级为一个显式的、有向无环图（DAG），或者更准确地说，一个**状态机**。
    -   **节点（Node）**：图中的每个节点代表一个计算步骤（例如，一次LLM调用，一次工具执行）。
    -   **边（Edge）**：节点之间的边代表控制流。我们可以定义条件边，根据当前状态决定下一步走向何方（例如，如果工具执行成功，则进入A节点；如果失败，则进入B节点进行重试）。
    -   **状态（State）**：整个图共享一个中心化的状态对象，每个节点都可以读取和修改它。这使得追踪和管理Agent的生命周期变得极其容易。
    **一句话定位**：LangGraph是实现本书核心理念——**可控性**——的最关键工具。

-   **`Pydantic`：Agent的“数据契约”**
    Pydantic本身不是一个AI框架，但它在构建可靠的Agent中扮演着至关重要的角色。它通过Python的类型提示，为我们提供了强大的数据验证和设置管理能力。
    -   **定义工具接口**：我们可以用Pydantic模型清晰地定义一个工具需要哪些参数、参数的类型和描述。这使得LLM能更准确地理解和调用我们的工具。
    -   **定义状态对象**：LangGraph中的中心状态对象，通常就是一个Pydantic模型，确保了状态数据的类型安全和可预测性。
    -   **定义Handoff协议**：在多Agent协作时，我们可以用Pydantic模型来定义Agent之间传递任务的“数据契约”，确保信息传递的准确无误。
    **一句话定位**：Pydantic是Agent系统中不同组件之间沟通的“通用语言”和“接口规范”。

---

## 2.3 实战Lab：构建第一个“Hello, World!”Agent

现在，让我们整合所学，构建第一个Agent。这个Agent的任务很简单：**查询指定城市的天气**。我们将为此定义一个（模拟的）工具，并观察Agent如何思考并调用它。

在你的项目根目录下，创建一个名为 `chapter_02_hello_agent.py` 的文件。

```python
import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

# --- 1. 定义工具 --- #
# 我们使用 @tool 装饰器来定义一个LangChain工具
# Pydantic风格的类型提示会自动被转换为LLM可理解的JSON Schema
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    print(f"---> 工具被调用：获取 {city} 的天气")
    if city == "北京":
        return "北京今天晴，25摄氏度。"
    elif city == "上海":
        return "上海今天有小雨，20摄氏度。"
    else:
        return f"抱歉，我不知道 {city} 的天气。"

tools = [get_weather]

# --- 2. 定义状态 --- #
# StateGraph 的状态必须是一个 TypedDict
# 这定义了我们Agent工作流中所有节点共享的状态对象
class AgentState(TypedDict):
    # `messages` 字段将包含整个对话历史
    # `Annotated` 的最后一个元素 "operator.add" 意味着每次都将新消息追加到列表中，而不是替换
    messages: Annotated[list, lambda x, y: x + y]

# --- 3. 定义图的节点 --- #

# 初始化模型。我们将使用DeepSeek，它与OpenAI的API兼容
# 我们在.env文件中配置了OPENAI_API_KEY和OPENAI_API_BASE
model = ChatOpenAI(temperature=0)

# 将模型绑定工具，这样模型就知道它可以调用哪些工具
bind_model = model.bind_tools(tools)

def agent_node(state: AgentState):
    """Agent节点：调用LLM，决定下一步行动"""
    print("--- Agent 节点 --- ")
    # 调用LLM，传入当前所有消息
    response = bind_model.invoke(state["messages"])
    # 将LLM的响应（可能是一个ToolCall）作为新消息返回
    return {"messages": [response]}

def tool_node(state: AgentState):
    """工具节点：如果LLM决定调用工具，则在此执行"""
    print("--- 工具 节点 ---")
    # 获取上一条消息，也就是LLM生成的ToolCall
    last_message = state["messages"][-1]
    
    # 遍历所有工具调用请求
    tool_calls = last_message.tool_calls
    
    # 执行工具并收集结果
    tool_messages = []
    for call in tool_calls:
        print(f"执行工具调用: {call['name']}({call['args']})")
        # 调用工具函数
        response = get_weather.invoke(call["args"])
        # 将工具的输出封装成 ToolMessage
        tool_messages.append(ToolMessage(content=str(response), tool_call_id=call["id"]))
    
    # 将工具结果作为新消息返回
    return {"messages": tool_messages}

# --- 4. 定义图的边 --- #

def router(state: AgentState) -> str:
    """路由器：根据最新消息的类型，决定下一个节点"""
    print("--- 路由器 --- ")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        # 如果有工具调用请求，则走向工具节点
        print("决策: 调用工具")
        return "tool_node"
    else:
        # 否则，结束流程
        print("决策: 结束")
        return "__end__"

# --- 5. 构建图 --- #

# 定义一个新的状态图
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("agent", agent_node)
graph.add_node("tool_node", tool_node)

# 设置入口点
graph.set_entry_point("agent")

# 添加条件边
graph.add_conditional_edges(
    "agent",
    router,
    {
        "tool_node": "tool_node",
        "__end__": END
    }
)

# 从工具节点返回到Agent节点，形成循环
graph.add_edge("tool_node", "agent")

# 编译成可执行的Runnable对象
runnable = graph.compile()

# --- 6. 运行Agent --- #

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    # 定义初始输入
    initial_input = {"messages": [HumanMessage(content="北京今天天气怎么样？")]}
    
    print("--- 开始执行 Agent ---")
    # 使用 stream 模式来逐步获取并打印每一步的状态
    for step in runnable.stream(initial_input):
        # step 是一个字典，key是节点名，value是该节点输出的状态
        node_name = list(step.keys())[0]
        node_output = step[node_name]
        print(f"\n<<< {node_name} >>>")
        print(f"{node_output}")
    print("--- Agent 执行结束 ---")

```

### 2.3.1 运行与解读

在你的VSCode终端中（它已经在Dev Container里了），运行这个Python文件：

```bash
python chapter_02_hello_agent.py
```

你将会看到类似下面的输出，它清晰地展示了Agent的每一步思考和行动：

```
--- 开始执行 Agent ---

<<< agent >>>
{'messages': [AIMessage(content='', tool_calls=[{'name': 'get_weather', 'args': {'city': '北京'}, 'id': 'call_abc'}])]}
--- 路由器 ---
决策: 调用工具

<<< tool_node >>>
--- 工具 节点 ---
执行工具调用: get_weather({'city': '北京'})
---> 工具被调用：获取 北京 的天气
{'messages': [ToolMessage(content='北京今天晴，25摄氏度。', tool_call_id='call_abc')]}

<<< agent >>>
--- Agent 节点 ---
{'messages': [AIMessage(content='北京今天天气晴朗，气温为25摄氏度。')]}
--- 路由器 ---
决策: 结束

<<< __end__ >>>
{'messages': [HumanMessage(content='北京今天天气怎么样？'), AIMessage(content='', tool_calls=[...]), ToolMessage(content='北京今天晴，25摄氏度。'), AIMessage(content='北京今天天气晴朗，气温为25摄氏度。')]}
--- Agent 执行结束 ---
```

**输出解读**：

1.  **第一次`agent`节点**：Agent接收到人类问题后，LLM正确地判断出需要调用`get_weather`工具，并生成了相应的`tool_calls`。
2.  **`router`决策**：路由器检查到`tool_calls`的存在，将流程导向`tool_node`。
3.  **`tool_node`执行**：`get_weather`函数被实际执行，其返回的字符串被包装成`ToolMessage`。
4.  **第二次`agent`节点**：`ToolMessage`被重新提交给LLM。这一次，LLM看到了工具的执行结果，于是它不再调用工具，而是生成了一句通顺的、总结性的自然语言回答。
5.  **`router`再次决策**：路由器检查到最新的消息中没有`tool_calls`，于是将流程导向`END`，执行结束。

这个简单的例子，完整地展示了一个基于`LangGraph`的Agent如何通过状态机实现“思考-行动”的循环，这是我们后续构建更复杂、更可控系统的基础。

---

## 2.4 本章小结

-   我们成功地使用**Docker和VSCode Dev Containers**搭建了一个隔离、一致的开发环境，这是专业软件工程的最佳实践。
-   我们厘清了三个核心框架的哲学定位：**LangChain**是基础组件库，**LangGraph**是实现可控性的状态机引擎，**Pydantic**是确保数据一致性的契约。
-   通过一个查询天气的实战Lab，我们构建并运行了第一个基于`LangGraph`的智能体，并详细解读了其内部的“思考-行动”循环，直观地理解了**Agenticness**是如何通过代码实现的。

在下一章，我们将深入探讨`LangGraph`的内部机制，学习如何通过`Hook`、`Wrap`、`Jump`等高级功能，为我们的Agent装上“仪表盘”和“方向盘”，正式开启**可控性**的探索之旅。
