# 第十章：`SkillCraft`框架设计：实现`ToolNode`与`Handoff`

欢迎来到本书的终极项目！从本章开始，我们将从一个“学习者”转变为一个“创造者”。我们的目标是亲手构建一个我们自己的、微型但功能完备的智能体框架——`SkillCraft`。这个框架的核心思想，正是对**第七章**中**ToolNode总线**和**Handoff协议**的工程化实现。

在本章，我们将专注于`SkillCraft`框架的基石：**技能的定义与发现**。我们将创建一个`@skill`装饰器，让开发者能轻易地将任何Python函数“注册”成一个可被AI理解的结构化“技能”；我们还将创建一个`SkillRegistry`，它像一个技能总线，负责自动发现和管理这些技能。

---

## 10.1 目标：将`ToolNode`和`Handoff`思想框架化

让我们回顾一下第七章的核心思想：

-   **ToolNode总线**：将所有工具注册到一个中心化的、可审计的节点上。
-   **Handoff协议**：定义清晰的数据契约，用于在不同Agent或组件之间交接任务。

`SkillCraft`框架的目标，就是将这两个思想，从一种“模式”或“约定”，提升为一个具体的、可供开发者直接使用的“框架”。

-   `@skill`装饰器 + `SkillRegistry`  ->  **ToolNode总线的实现**
-   使用Pydantic模型作为技能的参数  ->  **Handoff协议的实现**

当一个LLM决定调用一个`@skill`装饰的函数时，它需要返回一个包含技能名称和参数的结构化数据（如JSON），这本身就是一次清晰的“任务交接”。我们的框架将负责解析这个数据，并执行对应的Python函数。

---

## 10.2 `@skill`装饰器的实现：Python的“语法糖”魔法

我们的第一个目标是创建一个`@skill`装饰器。这个装饰器需要完成三件事：

1.  接收关于这个技能的元数据：**名称（name）**、**描述（description）**。
2.  接收一个Pydantic模型作为**参数规范（params）**。
3.  将以上信息，连同被装饰的函数本身，打包成一个结构化的`SkillDefinition`对象，并注册到一个全局的注册表中。

### 10.2.1 代码实现

让我们创建一个名为 `skillcraft` 的新目录，并在其中创建 `core.py` 文件。

```python
# skillcraft/core.py

import inspect
from typing import Callable, Dict, Type
from pydantic import BaseModel

# 全局的技能注册表，一个简单的字典
# 键是技能名，值是SkillDefinition对象
_GLOBAL_SKILL_REGISTRY: Dict[str, "SkillDefinition"] = {}

class SkillDefinition:
    """用于封装一个技能的完整定义。"""
    def __init__(self, name: str, description: str, params_cls: Type[BaseModel], func: Callable):
        self.name = name
        self.description = description
        self.params_cls = params_cls
        self.func = func

    def to_openai_tool(self) -> Dict:
        """将技能定义转换为OpenAI Tools格式的JSON Schema。"""
        # Pydantic的 schema() 方法可以自动生成JSON Schema
        schema = self.params_cls.schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }

def skill(name: str, description: str, params: Type[BaseModel]) -> Callable:
    """`@skill` 装饰器，用于定义并注册一个技能。"""
    def decorator(func: Callable):
        # 检查被装饰的是否为函数
        if not inspect.isfunction(func):
            raise TypeError("被@skill装饰的对象必须是一个函数。")

        # 创建技能定义实例
        skill_def = SkillDefinition(
            name=name,
            description=description,
            params_cls=params,
            func=func
        )

        # 注册到全局注册表
        if name in _GLOBAL_SKILL_REGISTRY:
            print(f"警告：技能 '{name}' 已被重新定义。")
        _GLOBAL_SKILL_REGISTRY[name] = skill_def
        
        print(f"--- 技能已注册: {name} ---")
        return func
    return decorator

```

**代码解读**：

-   我们创建了一个全局字典 `_GLOBAL_SKILL_REGISTRY` 作为我们的技能注册中心。
-   `SkillDefinition` 类是一个数据容器，它保存了一个技能的所有信息，最关键的是它有一个`to_openai_tool`方法，可以将Pydantic模型转换成LLM（如OpenAI, DeepSeek）在“工具使用”模式下所期望的JSON Schema格式。
-   `@skill`装饰器接收元数据，在内部创建`SkillDefinition`对象，并将其存入全局注册表。这是一个典型的**注册模式**。

---

## 10.3 `SkillRegistry`的实现：技能的“总线”与“清单”

现在我们有了一个注册表，我们还需要一个管理者。`SkillRegistry`类将扮演这个角色，它负责：

1.  提供访问已注册技能的方法。
2.  生成一份格式化的“技能清单”，用于注入到LLM的Prompt中，告诉LLM它有哪些能力。

### 10.3.1 代码实现

继续在 `skillcraft/core.py` 文件中添加以下代码：

```python
# skillcraft/core.py (续)

class SkillRegistry:
    """技能注册表的管理者。"""

    def get_skill(self, name: str) -> SkillDefinition:
        """根据名称获取一个已注册的技能。"""
        skill_def = _GLOBAL_SKILL_REGISTRY.get(name)
        if not skill_def:
            raise ValueError(f"技能 '{name}' 未找到。")
        return skill_def

    def get_all_skills(self) -> Dict[str, SkillDefinition]:
        """获取所有已注册的技能。"""
        return _GLOBAL_SKILL_REGISTRY

    def get_openai_tools(self) -> list[Dict]:
        """获取所有技能的OpenAI Tools格式定义列表。"""
        return [skill.to_openai_tool() for skill in _GLOBAL_SKILL_REGISTRY.values()]

    def execute_skill(self, name: str, params: Dict) -> any:
        """执行一个技能。"""
        skill_def = self.get_skill(name)
        
        # 使用Pydantic模型来验证和解析参数
        try:
            validated_params = skill_def.params_cls(**params)
        except Exception as e:
            raise ValueError(f"为技能 '{name}' 提供的参数无效: {e}")
        
        # 调用原始的Python函数
        return skill_def.func(validated_params)

```

**代码解读**：

-   `get_skill` 和 `get_all_skills` 提供了对注册表的只读访问。
-   `get_openai_tools` 是一个关键的辅助方法。它遍历所有技能，调用它们的`to_openai_tool`方法，生成一个可以直接传递给`model.bind_tools()`的列表。这极大地简化了将技能“告知”LLM的过程。
-   `execute_skill` 是我们的**ToolNode总线**的核心执行逻辑。它接收技能名称和LLM返回的参数字典，先用Pydantic模型进行严格的**类型验证**，然后才调用真正的Python函数。这确保了每次函数调用都是类型安全的。

---

## 10.4 实战：定义第一个技能并验证框架

现在，`SkillCraft`框架的基石已经完成。让我们来实际使用它，定义一个简单的技能，并验证整个注册和转换流程是否正常工作。

创建一个新文件 `chapter_10_test_skillcraft.py`。

```python
# chapter_10_test_skillcraft.py

import json
from pydantic import BaseModel, Field
from skillcraft.core import skill, SkillRegistry

# --- 1. 定义技能所需的参数模型 --- #
class GetWeatherParams(BaseModel):
    city: str = Field(..., description="需要查询天气的城市名，例如 '北京'。")

# --- 2. 使用 @skill 装饰器定义一个技能 --- #
@skill(
    name="get_weather",
    description="获取指定城市的实时天气信息。",
    params=GetWeatherParams
)
def get_weather(params: GetWeatherParams) -> str:
    """一个获取天气信息的技能实现。"""
    city = params.city
    print(f"---> 技能被执行：获取 {city} 的天气")
    if city == "北京":
        return "北京今天晴，25摄氏度。"
    elif city == "上海":
        return "上海今天有小雨，20摄氏度。"
    else:
        return f"抱歉，我不知道 {city} 的天气。"

# --- 3. 验证 SkillCraft 框架 --- #
if __name__ == "__main__":
    # 实例化技能注册表管理者
    registry = SkillRegistry()

    # 验证技能是否已注册
    print("--- 已注册的技能 ---")
    all_skills = registry.get_all_skills()
    print(list(all_skills.keys()))

    # 验证OpenAI Tools格式的生成
    print("\n--- 生成的OpenAI Tool Schema ---")
    openai_tools_schema = registry.get_openai_tools()
    # 使用json.dumps美化打印
    print(json.dumps(openai_tools_schema, indent=2, ensure_ascii=False))

    # 验证技能的执行
    print("\n--- 测试技能执行 ---")
    # 模拟LLM返回的参数
    mock_llm_params = {"city": "上海"}
    result = registry.execute_skill("get_weather", mock_llm_params)
    print(f"技能执行结果: {result}")

    # 测试无效参数
    print("\n--- 测试无效参数 ---")
    try:
        registry.execute_skill("get_weather", {"location": "北京"}) # 错误的参数名
    except ValueError as e:
        print(f"成功捕获到错误: {e}")

```

运行 `python chapter_10_test_skillcraft.py`，你将看到框架按预期工作：技能被成功注册，生成了正确的JSON Schema，并且能够安全地执行技能并对无效参数抛出错误。

---

## 10.5 本章小结

-   本章我们迈出了构建`SkillCraft`框架的第一步，也是最重要的一步。我们成功地将**第七章**中**ToolNode总线**和**Handoff协议**的理论思想，物化为了具体的代码实现。
-   我们创建了 **`@skill` 装饰器**，它提供了一种极其简洁和声明式的方式来定义一个结构化的技能，将业务逻辑与框架注册逻辑分离。
-   我们实现了 **`SkillRegistry` 类**，它作为技能的“总线”和“管理者”，负责技能的存储、发现、格式转换和安全执行。
-   `SkillRegistry`的`execute_skill`方法，通过Pydantic模型的验证，为我们的“Handoff协议”提供了**类型安全**的保障。
-   `SkillRegistry`的`get_openai_tools`方法，为我们将技能“告知”LLM提供了巨大的便利，是实现**技能发现**的关键。

我们已经打造好了`SkillCraft`框架的“骨架”。在下一章，我们将为它注入“灵魂”——创建一个`OrchestratorAgent`，这个Agent将能够理解我们定义的技能，并智能地选择和调用它们来完成任务。我们将正式进入**`ACE/Manus`**思想的工程实践。
