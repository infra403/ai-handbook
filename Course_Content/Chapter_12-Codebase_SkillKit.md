# 第十二章：实战应用（一）：使用`SkillCraft`打造可进化的`Codebase SkillKit`

在过去的十一章里，我们一直在扮演“框架设计师”和“架构师”的角色。我们从理论出发，一步步构建了`SkillCraft`框架的骨架和灵魂。现在，我们将转换角色，成为这个框架的**第一批用户**。我们将亲身体验，使用我们自己创造的工具，来打造一个真正有价值的资产——一个可复用、可分发的**代码库技能包（`Codebase SkillKit`）**。

本章的目标是“吃自己的狗粮”（Dogfooding）。我们将利用`@skill`装饰器，将一系列复杂的代码操作，封装成一个个独立的、语义清晰的技能。在这个过程中，我们会将**第四章**的**Checkpoint**和**第五章**的**RAG**等核心技术，作为实现这些高级技能的“内部逻辑”，从而真正体现本书知识体系的融会贯通。

---

## 12.1 “吃自己的狗粮”：成为自己框架的用户

“吃自己的狗粮”是软件工程中的一句谚语，指的是公司（或团队）在公开发布产品前，自己先在日常工作中使用它。这是发现产品缺陷、验证产品价值的最佳方式。

通过在本章使用`SkillCraft`框架，我们将深刻体会到：

-   **抽象的价值**：一个设计良好的框架，能将复杂的底层逻辑（如状态管理、工具调用、错误处理）封装起来，让用户可以专注于实现业务功能（技能本身）。
-   **声明式编程的优雅**：我们只需要用`@skill`来“声明”一个技能的元数据和参数，而无需关心它如何被发现、如何被调用。
-   **可控性的内建**：由于我们的`OrchestratorAgent`内置了中间件和Checkpoint机制，我们创建的任何技能，都将自动地、免费地享受到这些可控性保障。

---

## 12.2 实现`code_refactor`技能：融合Checkpoint的“安全写入”

我们的第一个，也是最重要的技能，是`code_refactor`。它的任务是根据指令修改一个文件。这是一个典型的高风险操作，因此，它的内部实现必须是**健壮和安全的**。

我们将在这里应用**第四章**的**Checkpoint**思想，但不是在Agent层面，而是在**技能内部**，实现一个“原子化”的文件写入操作：**要么成功，要么回滚**。

### 12.2.1 代码实现

创建一个新文件 `skill_kits/codebase.py`。

```python
# skill_kits/codebase.py

import os
import shutil
from pydantic import BaseModel, Field
from skillcraft.core import skill

# --- 1. 定义参数模型 --- #
class CodeRefactorParams(BaseModel):
    file_path: str = Field(..., description="需要重构的文件的完整路径。")
    refactor_instruction: str = Field(..., description="对代码进行重构的具体指令。")
    # 假设LLM会生成完整的、用于替换的新代码
    new_code: str = Field(..., description="由LLM生成、用于完整替换旧文件内容的新代码。")

# --- 2. 定义技能 --- #
@skill(
    name="code_refactor",
    description="安全地重构一个代码文件。它会先备份原文件，写入新代码，如果后续步骤（如测试）失败，可以进行回滚。",
    params=CodeRefactorParams
)
def code_refactor(params: CodeRefactorParams) -> str:
    """安全地重构一个代码文件，包含备份和回滚逻辑。"""
    backup_path = params.file_path + ".bak"
    
    # --- Checkpoint思想的应用 --- #
    try:
        # 1. 创建“快照”：备份原始文件
        print(f"---> [code_refactor] 备份原始文件到: {backup_path}")
        if os.path.exists(params.file_path):
            shutil.copy(params.file_path, backup_path)
        
        # 2. 执行核心操作：写入新代码
        print(f"---> [code_refactor] 正在写入新代码到: {params.file_path}")
        with open(params.file_path, "w", encoding="utf-8") as f:
            f.write(params.new_code)
            
        # 假设这个技能只是写入，后续由另一个`run_tests`技能来验证
        # 如果测试失败，OrchestratorAgent可以决定调用一个`rollback_refactor`技能
        return f"文件 '{params.file_path}' 已被成功重构。备份文件位于 '{backup_path}'。"

    except Exception as e:
        print(f"错误：在重构文件时发生异常: {e}")
        # 简单的回滚逻辑
        if os.path.exists(backup_path):
            print(f"---> [code_refactor] 发生错误，正在从备份中回滚... ")
            shutil.move(backup_path, params.file_path)
        return f"文件重构失败: {e}"

# 我们可以再定义一个专门的回滚技能，让Agent可以显式调用
class RollbackParams(BaseModel):
    original_path: str = Field(..., description="原始文件的路径。")

@skill(
    name="rollback_refactor",
    description="当代码重构或修改导致测试失败时，从备份文件恢复原始代码。",
    params=RollbackParams
)
def rollback_refactor(params: RollbackParams) -> str:
    backup_path = params.original_path + ".bak"
    if not os.path.exists(backup_path):
        return f"错误：找不到备份文件 '{backup_path}'。无法回滚。"
    
    print(f"---> [rollback_refactor] 正在从 '{backup_path}' 回滚... ")
    shutil.move(backup_path, params.original_path)
    return f"文件 '{params.original_path}' 已成功回滚。"
```

**设计思想解读**：

-   `code_refactor`技能的职责非常单一：就是“安全地写入”。它通过**备份文件**这一简单而有效的方式，实现了**Checkpoint**的核心思想。
-   我们将**回滚**操作也封装成了一个独立的`rollback_refactor`技能。这使得我们的`OrchestratorAgent`在“自修正”阶段，有了一个明确的、可以调用的原子操作，而不是在复杂的代码中去实现回滚逻辑。

---

## 12.3 实现`smart_code_reader`技能：应用RAG进行精准读取

在处理大型代码库时，Agent不可能一次性读取所有文件。它需要一个“智能”的读取器，能够根据任务描述，只读取最相关的代码片段。这正是**第五章**中**RAG**思想的应用场景。

`smart_code_reader`技能将接收一个高阶的任务描述，然后自己负责**检索（Select）**相关文件和代码块，并将它们作为上下文返回，供其他技能或Agent使用。

### 12.3.1 代码实现

```python
# skill_kits/codebase.py (续)

# ... (其他imports)
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import glob

# --- 1. 定义参数模型 --- #
class SmartCodeReaderParams(BaseModel):
    task_description: str = Field(..., description="关于当前任务的详细描述，例如 '修复用户登录时的密码验证bug'。")
    file_glob_pattern: Optional[str] = Field(None, description="一个可选的glob模式，用于缩小文件搜索范围，例如 'src/**/*.py'。")

# --- 2. 定义技能 --- #
@skill(
    name="smart_code_reader",
    description="智能代码阅读器。根据任务描述，自动查找并读取最相关的代码文件和片段，为后续的代码编写或分析提供上下文。",
    params=SmartCodeReaderParams
)
def smart_code_reader(params: SmartCodeReaderParams) -> str:
    """应用RAG思想，实现一个智能代码阅读器。"""
    print(f"---> [smart_code_reader] 收到任务: {params.task_description}")
    
    # 1. 查找文件 (简单的文件发现)
    if params.file_glob_pattern:
        all_files = glob.glob(params.file_glob_pattern, recursive=True)
    else: # 默认在当前目录下查找
        all_files = glob.glob("**/*.py", recursive=True)
    
    if not all_files:
        return "错误：找不到任何匹配的代码文件。"

    # 2. 读取和切分代码 (Write & Split)
    # 这里我们为Python代码使用专门的切分器
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    docs = []
    for f_path in all_files:
        try:
            with open(f_path, "r") as f:
                code = f.read()
            for chunk in splitter.create_documents([code], metadatas=[{"source": f_path}]):
                docs.append(chunk)
        except Exception:
            continue # 忽略无法读取的文件

    # 3. 构建内存中的向量数据库 (Embed & Store)
    # 注意：这是一个开销较大的操作，真实应用中会使用持久化的数据库
    print(f"---> [smart_code_reader] 正在为 {len(docs)} 个代码块创建向量索引...")
    vector_store = Chroma.from_documents(docs, OpenAIEmbeddings())
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # 返回最相关的5个块

    # 4. 检索最相关的代码块 (Select)
    print(f"---> [smart_code_reader] 正在检索与任务最相关的代码...")
    relevant_chunks = retriever.invoke(params.task_description)

    # 5. 格式化并返回结果
    context_str = """
以下是根据您的任务描述，找到的最相关的代码片段：

"""
    for chunk in relevant_chunks:
        context_str += f"--- 来自文件: {chunk.metadata['source']}\n---\n```python\n{chunk.page_content}\n```\n\n"
    
    return context_str
```

**设计思想解读**：

-   这个技能完美地复刻了**第五章**的RAG流程，但它的应用场景从“问答”变成了“为编程任务提供上下文”。
-   它将复杂的RAG逻辑（加载、切分、嵌入、检索）封装在一个单一的、语义清晰的`smart_code_reader`技能中。
-   现在，我们的`OrchestratorAgent`在接到一个编程任务时，它的**规划（Plan）**的第一步就可以是：“调用`smart_code_reader`来理解当前代码库的相关部分”。这极大地提升了Agent的效率和智能性。

---

## 12.4 `[新增]` 实现`learn_from_feedback`技能：构建进化能力

为了让我们最终的AI代码助手能够“越用越聪明”，我们需要为它实现一个学习机制。这就是**第四章**中“从短期记忆到长期知识”思想的落地。

我们将创建一个`learn_from_feedback`技能，它负责将一次成功的“人机协同”经验，存入一个长期的知识库。

```python
# skill_kits/codebase.py (续)

# 假设我们有一个全局的、持久化的向量数据库用于存储经验
# 在真实应用中，它应该在程序启动时加载
experience_db = Chroma(persist_directory="./experience_db", embedding_function=OpenAIEmbeddings())

class LearnFromFeedbackParams(BaseModel):
    task: str = Field(..., description="原始的任务描述。")
    failed_attempt: str = Field(..., description="Agent失败的尝试或方案。")
    human_guidance: str = Field(..., description="人类提供的、用于解决问题的正确指导或代码片段。")

@skill(
    name="learn_from_feedback",
    description="学习一次成功的‘人机协同’经验，将其存入长期记忆，以便未来遇到类似问题时可以参考。",
    params=LearnFromFeedbackParams
)
def learn_from_feedback(params: LearnFromFeedbackParams) -> str:
    """将一次成功的反馈学习经验存入向量数据库。"""
    experience_text = f"""## 经验案例
### 原始任务:
{params.task}

### 失败的尝试:
{params.failed_attempt}

### 成功的指导方案 (由人类提供):
{params.human_guidance}
"""
    
    print("---> [learn_from_feedback] 正在学习新的经验... ")
    experience_db.add_texts([experience_text])
    experience_db.persist()
    
    return "经验已学习并存入长期记忆。"
```

**设计思想解读**：

-   这个技能为Agent的**进化**提供了一个具体的API。
-   在**第十一章**的`OrchestratorAgent`中，当它在`AWAIT_FEEDBACK`节点成功获得人类指导并最终完成任务后，它的最后一步就应该是调用`learn_from_feedback`技能，将这次宝贵的经验记录下来。
-   相应地，`OrchestratorAgent`的`PLAN`节点也需要升级：在制定计划前，先用原始任务描述去**检索`experience_db`**。如果找到了相关的历史经验，就将这个经验一并提供给LLM，辅助它做出更高质量的规划。

---

## 12.5 打包与分发

现在，我们的`skill_kits/codebase.py`文件中已经包含了一组强大的、可复用的技能。为了让其他项目或Agent也能使用它们，我们可以简单地将其打包成一个Python包。这通常涉及到创建一个`pyproject.toml`或`setup.py`文件，这超出了本书的核心范围，但其思想是通用的：**将一组相关的技能组织成一个独立的、可安装的“技能包”**。这就是通往“技能生态”的第一步。

## 12.6 本章小结

-   本章我们转换角色，作为`SkillCraft`框架的用户，成功地“吃自己的狗粮”，验证了框架的价值。
-   我们开发了一个`Codebase SkillKit`，其中包含了三个核心的高级技能：
    1.  **`code_refactor`**：它将**第四章**的**Checkpoint**思想应用于技能内部，实现了安全的、可回滚的文件操作。
    2.  **`smart_code_reader`**：它将**第五章**的**RAG**思想应用于代码理解，实现了为复杂任务精准提供上下文的能力。
    3.  **`learn_from_feedback`**：它将**第四章**的“长期知识”思想落地，为Agent的**反馈进化**提供了具体的实现机制。
-   通过本章的实践，我们深刻地理解了如何将基础的“可控性”技术，作为构建高级、安全、智能的“代理性”能力的内部组件。

我们已经拥有了强大的框架和一套专业的技能。在最后一章，我们将把它们组装起来，完成我们的最终作品——一个完整的、可进化的AI代码助手。
