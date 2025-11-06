# 第十一章：`OrchestratorAgent`：实现`ACE/Manus`的“规划-执行-修正”引擎

在第十章，我们成功打造了`SkillCraft`框架的“骨架”——一个强大的技能定义与注册中心。现在，是时候为这个骨架注入“灵魂”了。这个灵魂，就是一个能够理解并智能地编排、调用这些技能的“大脑”——**`OrchestratorAgent`（编排智能体）**。

本章的目标，是将**第六章**中探讨的`ACE/Manus/DeepAgents`等高级代理思想，真正地通过代码工程化。我们将不再满足于简单的`ReAct`循环，而是要构建一个具备**“规划-执行-修正”**能力的、更高级的认知引擎。同时，我们还会将**第八章**的**中间件**思想融入其中，为这个强大的引擎装上“安全阀”。

---

## 11.1 架构升级：从`ReAct`到“规划-执行-修正”

让我们再次明确架构的升级。一个简单的`ReAct` Agent，其心智模型是线性的：“我应该采取什么**下一步**行动？”。而一个`OrchestratorAgent`，其心智模型是战略性的、分层的：

1.  **规划（Plan）**：面对一个复杂目标，它首先自问：“要完成这个目标，我需要执行一个怎样的**步骤序列**？”
2.  **执行（Execute）**：它从计划中取出第一步，并思考：“要完成**这一步**，我应该调用哪个技能？”（这是一个微型的`ReAct`）。
3.  **评估（Evaluate）**：执行完一步后，它会审视结果，并自问：“这一步的结果是否符合预期？整个计划是否还需要继续？”
4.  **修正（Self-Correct）**：如果结果不符合预期，它会进入最关键的阶段，自问：“基于失败的反馈，我应该如何**修改我的计划**以最终达成目标？”

这个`PLAN -> EXECUTE -> EVALUATE -> (optional) SELF_CORRECT`的循环，是高级代理性的核心体现。我们将使用`LangGraph`来精确地将这个循环建模为一个状态图。

---

## 11.2 `LangGraph`实现：构建认知循环的状态图

我们将设计一个包含`PLAN`、`EXECUTE_STEP`、`EVALUATE_RESULT`和`SELF_CORRECT`等核心节点的状态图。此外，我们还将引入一个`AWAIT_FEEDBACK`节点，用于实现我们的人机协同机制。

### 11.2.1 状态定义

首先，我们需要一个更丰富的状态对象来承载这个复杂的循环。

```python
# skillcraft/orchestrator.py (新文件)

from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage

class OrchestratorState(TypedDict):
    # 对话历史
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    # Agent生成的执行计划
    plan: Optional[list[str]]
    # 已完成的步骤
    past_steps: Annotated[list[tuple[str, str]], lambda x, y: x + y]
    # 当前步骤的执行结果
    current_step_result: Optional[str]
    # 原始用户请求
    original_request: str
```

### 11.2.2 节点实现（伪代码）

让我们用伪代码来勾勒出每个节点的核心逻辑。

-   **`PLAN` 节点**：
    -   接收`original_request`。
    -   调用LLM，生成一个步骤列表，存入`plan`字段。

-   **`EXECUTE_STEP` 节点**：
    -   从`plan`和`past_steps`中确定当前要执行的步骤。
    -   调用LLM（绑定了`SkillRegistry`中的所有技能），让它根据当前步骤的指令，选择一个技能并返回其`ToolCall` JSON。
    -   **注意**：这里不直接执行，而是将`ToolCall`作为消息返回。这给了我们干预的机会。

-   **`TOOL_EXECUTION` 节点**：
    -   接收`ToolCall`消息。
    -   **（注入可控性）** 在这里，我们可以调用**中间件**，例如`HumanApprovalMiddleware`。
    -   如果中间件允许，则调用`SkillRegistry.execute_skill()`来实际执行技能。
    -   将执行结果存入`current_step_result`。

-   **`EVALUATE_RESULT` 节点**：
    -   查看`current_step_result`。
    -   判断当前步骤是否成功，计划是否需要继续。
    -   **路由逻辑**：决定下一步是`EXECUTE_STEP`（继续计划）、`SELF_CORRECT`（执行失败）还是`END`（计划完成）。

-   **`SELF_CORRECT` 节点**：
    -   收集`original_request`, `plan`, `failed_step`, `error_message`。
    -   调用LLM，生成一个**新的、修正过的计划**。
    -   用新计划覆盖`plan`字段，并清空`past_steps`。

-   **`AWAIT_FEEDBACK` 节点**：
    -   当`SELF_CORRECT`多次失败后进入此节点。
    -   向用户打印求助信息，并暂停图的执行，等待人类输入。

---

## 11.3 `[新增]` 融入经济学与人机协同

现在，我们将之前讨论的两个高级概念融入这个架构中。

-   **经济学（成本估算）**：在`PLAN`节点，我们可以让LLM在生成计划时，为每个步骤标注一个预估的“成本”或建议使用的模型（如`'step': '...', 'model': 'gpt-4-turbo'` vs. `'model': 'gpt-3.5-turbo'`）。在`EXECUTE_STEP`节点中，我们会根据这个标注来选择不同的LLM实例。这为实现`BudgetControlMiddleware`提供了基础。

-   **人机协同（反馈循环）**：`AWAIT_FEEDBACK`节点是实现人机协同的关键。当Agent“卡住”时，它不是简单地放弃，而是优雅地将控制权交还给人类，请求指导。这使得整个系统更加健壮和实用。

---

## 11.4 注入可控性：实现`HumanApprovalMiddleware`

这是**第八章**中间件思想的直接应用。我们将实现一个可以在`TOOL_EXECUTION`节点中被调用的中间件，用于在执行高风险技能前获得人类批准。

```python
# skillcraft/middleware.py (新文件)

class HumanApprovalMiddleware:
    """一个在执行前请求人类批准的中间件。"""
    def __init__(self, high_risk_skills: list[str]):
        self.high_risk_skills = set(high_risk_skills)

    def __call__(self, skill_name: str, params: dict) -> bool:
        """如果技能是高风险的，则请求批准。返回True表示允许执行。"""
        if skill_name in self.high_risk_skills:
            print("\n--- 审批请求 ---")
            print(f"Agent 准备执行高风险技能: '{skill_name}'")
            print(f"参数: {params}")
            
            while True:
                response = input("是否批准执行？(yes/no): ").lower()
                if response in ["yes", "y"]:
                    print("--- 已批准 ---")
                    return True
                elif response in ["no", "n"]:
                    print("--- 已拒绝 ---")
                    return False
        # 对于非高风险技能，默认允许
        return True
```

在`TOOL_EXECUTION`节点中，我们只需在调用`registry.execute_skill`之前，调用这个中间件即可。

`if approval_middleware(skill_name, params): registry.execute_skill(...)`

---

## 11.5 实战：构建并验证`OrchestratorAgent`

由于完整的代码非常长，这里我们提供核心的构建和验证逻辑。

创建一个新文件 `chapter_11_orchestrator_test.py`。

```python
# chapter_11_orchestrator_test.py

# (这里会导入完整的OrchestratorState, 以及所有节点的实现)
# from skillcraft.orchestrator import create_orchestrator_graph
# from skillcraft.middleware import HumanApprovalMiddleware

# 假设 create_orchestrator_graph 已经根据上面的逻辑构建好了图

# 1. 定义一个会失败的技能，用于测试自修正
@skill(name="buggy_code_writer", ...)
def write_buggy_code(params):
    # ... 写入一段有明显bug的代码 ...
    return "代码已写入，但其中包含一个bug。"

@skill(name="test_runner", ...)
def run_tests(params):
    # ... 模拟运行测试，并总是返回失败 ...
    return "测试失败：TypeError: 'NoneType' object is not iterable."

# 2. 初始化中间件
approval_middleware = HumanApprovalMiddleware(high_risk_skills=["buggy_code_writer"])

# 3. 编译图，并注入中间件
# (在TOOL_EXECUTION节点中，我们会调用approval_middleware)
graph = create_orchestrator_graph(middleware=approval_middleware)
runnable = graph.compile(checkpointer=...)

# 4. 运行一个会失败的任务
if __name__ == "__main__":
    task = "请使用 buggy_code_writer 创建一个文件，然后用 test_runner 测试它。"
    initial_input = {"messages": [HumanMessage(content=task)], "original_request": task}
    config = {"configurable": {"thread_id": "test_self_correct_123"}}

    # 运行并观察
    for step in runnable.stream(initial_input, config):
        print(step)
        print("---")
```

**预期运行流程**：

1.  `PLAN`节点生成计划：`["call: buggy_code_writer", "call: test_runner"]`。
2.  `EXECUTE_STEP`节点准备执行`buggy_code_writer`。
3.  `TOOL_EXECUTION`节点调用`HumanApprovalMiddleware`，程序暂停，等待你在命令行输入`yes`。
4.  你输入`yes`后，技能被执行。
5.  `EVALUATE_RESULT`节点评估结果，继续计划。
6.  `EXECUTE_STEP`节点准备执行`test_runner`。
7.  `TOOL_EXECUTION`节点执行测试，返回失败结果。
8.  `EVALUATE_RESULT`节点检测到失败，将流程导向`SELF_CORRECT`节点。
9.  `SELF_CORRECT`节点接收到错误信息，生成一个新的、试图修复bug的计划。
10. 流程重新开始，体现了完整的“规划-执行-修正”循环。

---

## 11.6 本章小结

-   本章我们将**第六章**的`ACE/Manus`高级代理思想，成功地工程化为一个具备**“规划-执行-评估-修正”**能力的`OrchestratorAgent`。
-   我们详细设计了实现这一高级认知循环所需的`LangGraph`状态图，包括`PLAN`, `EXECUTE_STEP`, `EVALUATE_RESULT`, `SELF_CORRECT`等关键节点。
-   我们将在最终项目中实现**Agent经济学**（成本估算）和**人机协同**（`AWAIT_FEEDBACK`节点）等高级概念，让Agent更贴近生产实际。
-   我们亲手实现了**第八章**的**中间件**思想，创建了一个`HumanApprovalMiddleware`，并将其注入到`OrchestratorAgent`的执行流程中，为Agent的行动装上了“安全阀”。
-   通过一个实战验证计划，我们明确了如何测试Agent的规划、人工干预和自我修正能力。

至此，我们已经构建了`SkillCraft`框架的“骨架”（第十章）和“灵魂”（第十一章）。我们已经准备好用这个强大的框架，来打造我们的第一个真正的技能包了。
