# 第五章：上下文工程：Agent的“信息食谱”与“记忆防火墙”

在前面的章节中，我们的Agent已经具备了行动和记忆的能力。但随着对话的进行或任务的复杂化，一个新的、非常现实的问题会浮现出来：**上下文窗口（Context Window）**的限制。

大型语言模型（LLM）的上下文窗口，就像是它工作时的“短期记忆”或计算机的“RAM”。它是有限的，并且每一轮对话，我们都需要将全部历史消息和相关信息塞入这个窗口，这带来了几个严峻的挑战：

1.  **成本高昂**：上下文越长，API调用的Token消耗就越多，成本随之线性增长。
2.  **性能瓶颈**：更长的上下文通常意味着更长的模型响应时间。
3.  **“大海捞针”问题**：当上下文中充满了大量无关信息时，LLM可能难以找到解决当前问题的关键信息，导致其“注意力”分散，性能下降。

因此，**上下文工程**——即如何动态、智能地构建和管理输入给LLM的上下文——成为了构建高效、经济、强大的Agent的核心技术之一。本章，我们将学习如何为Agent的“注意力”构建一个强大的管理系统。

---

## 5.1 上下文治理：可控性的关键一环

**上下文治理**是**可控性**的一个重要延伸。它不仅仅是关于优化成本和性能，更是关于控制Agent在特定时刻“应该知道什么”和“不应该知道什么”。

> **上下文治理** 指的是通过一系列技术手段，精确地控制和塑造提供给LLM的上下文信息，以达到提升性能、降低成本、增强安全性、并引导Agent行为的目的。

一个优秀的上下文治理系统，应该像一个专业的图书管理员，当Agent需要某个知识时，他能迅速从浩如烟海的图书馆中，准确地找出那几本最相关的书，而不是把整个图书馆都搬到Agent面前。

这个“图书管理员”的核心能力，可以被分解为四个动作：**写入（Write）**、**选择（Select）**、**压缩（Compress）**和**隔离（Isolate）**。

---

## 5.2 RAG的四大核心操作

**检索增强生成（Retrieval-Augmented Generation, RAG）**是实现上下文治理最核心的技术框架。它将LLM从一个“封闭大脑”变成了一个能够主动利用外部知识的“开放大脑”。RAG的实现，正是围绕着这四大操作展开的。

### 5.2.1 写入（Write）：构建知识库

这是RAG的准备阶段，我们需要将Agent可能需要用到的外部知识，处理并存入一个专门的数据库（通常是**向量数据库**）中。

1.  **加载（Load）**：读取原始文档，如PDF、Markdown、HTML等。
2.  **切分（Split）**：将长文档切分成更小的、语义完整的块（Chunks）。这至关重要，因为我们后续是按块来检索的。一个好的切分策略（如按段落、按标题）能显著提升检索质量。
3.  **嵌入（Embed）**：使用一个**嵌入模型（Embedding Model）**将每个文本块转换成一个高维向量（Vector）。这个向量可以被认为是文本块在语义空间中的“坐标”。
4.  **存储（Store）**：将文本块和其对应的向量存入向量数据库中。

### 5.2.2 选择（Select）：精准的知识检索

当Agent收到一个问题时，它会执行“选择”操作，从知识库中找出最相关的信息。

1.  **查询向量化**：将用户的查询（问题）同样使用嵌入模型转换成一个查询向量。
2.  **相似度搜索**：在向量数据库中，计算查询向量与所有存储的文本块向量之间的“距离”或“相似度”（如余弦相似度）。
3.  **返回Top-K**：返回与查询最相似的K个文本块。这些文本块就是我们为Agent精心挑选的、用于构建上下文的“相关知识”。

### 5.2.3 压缩（Compress）：为上下文“瘦身”

即使是通过RAG检索到的知识，也可能存在冗余。**压缩**操作旨在进一步精简上下文，降低成本、提升效率。

-   **对话历史压缩**：对于非常长的对话，我们可以使用一个次级的、更快的LLM来对早期的对话历史进行总结，用一个简短的摘要来代替冗长的原文。
-   **文档压缩**：在将检索到的文档块放入上下文之前，可以先让LLM判断这个文档块与当前问题的相关性，并只提取出其中最关键的几句话。

### 5.2.4 隔离（Isolate）：构建信息的“沙箱”

在多租户或多任务场景下，确保不同上下文之间的信息不被泄露或干扰至关重要。这就是**隔离**的用武之地。

-   **按会话隔离**：在我们的Checkpointer机制中，使用`thread_id`就是一种天然的隔离。不同用户的对话历史被存储在不同的“档位”中。
-   **按任务隔离**：当一个Agent需要处理一个与主对话无关的子任务时，我们可以为这个子任务创建一个临时的、独立的上下文空间。子任务完成后，只将最终结果返回给主对话，而中间过程的细节则被丢弃，避免污染主对话的上下文。

---

## 5.3 实战Lab：构建一个RAG Agent

现在，让我们构建一个能回答特定文档内容的RAG Agent。我们将使用`langchain`内置的组件，并选择`ChromaDB`作为一个轻量级的、本地运行的向量数据库。

### 5.3.1 环境准备

1.  **更新Dockerfile**：我们需要安装`chromadb`和文档加载、切分所需的库。

    ```dockerfile
    # .devcontainer/Dockerfile
    RUN pip install --no-cache-dir \
        # ... (原有依赖)
        chromadb \
        langchain-text-splitters \
        pypdf
    ```
    修改后，请重建你的Dev Container。

2.  **准备知识文档**：在项目根目录创建一个名为 `knowledge.txt` 的文件，并填入一些内容。例如：

    ```text
    # knowledge.txt
    LangGraph是一个用于构建有状态、多角色应用的库，由LangChain团队开发。
    它的核心思想是将Agent的执行流程表示为一个图（Graph）。
    图中的节点（Node）代表计算单元，边（Edge）代表控制流。
    LangGraph通过Checkpoint机制实现持久化记忆。
    ```

### 5.3.2 编写代码

创建一个新文件 `chapter_05_rag_agent.py`。

```python
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# --- 1. 写入 (Write): 构建向量数据库 --- #

def build_vector_store():
    """加载文档，切分，并构建一个Chroma向量数据库。"""
    print("--- 开始构建向量数据库 ---")
    # 加载文档
    loader = TextLoader("knowledge.txt")
    documents = loader.load()

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(documents)

    # 嵌入并存储
    # OpenAIEmbeddings 会自动使用 .env 文件中的环境变量
    vectorstore = Chroma.from_documents(
        documents=all_splits, 
        embedding=OpenAIEmbeddings()
    )
    print("--- 向量数据库构建完成 ---")
    return vectorstore

# --- 2. 选择 (Select): 创建检索器 --- #

vectorstore = build_vector_store()
# 将向量数据库转换为一个检索器（Retriever）
# k=1 意味着每次只返回最相关的一个文档块
retriever = vectorstore.as_retriever(k=1)

# --- 3. 构建RAG链 --- #

# 定义提示模板
# {context} 将由检索到的文档内容填充
# {question} 将由用户的原始问题填充
template = """基于以下上下文来回答问题：
{context}

问题: {question}
"""
prompt = PromptTemplate.from_template(template)

# 初始化我们的大语言模型
model = ChatOpenAI(temperature=0)

# 创建RAG链 (Chain)
# 这就是LangChain的链式组合（LCEL）的威力
rag_chain = (
    # RunnableParallel允许我们并行执行检索和传递问题
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt
    | model
    | StrOutputParser() # 将模型的输出（AIMessage）解析为字符串
)

# --- 4. 运行RAG Agent --- #

if __name__ == "__main__":
    print("\n--- RAG Agent 已准备就绪，请输入您的问题 --- (输入'exit'退出)")
    while True:
        question = input("> ")
        if question.lower() == 'exit':
            break
        
        # 调用RAG链来获取答案
        answer = rag_chain.invoke(question)
        
        print(f"AI: {answer}\n")
```

### 5.3.3 运行与解读

运行 `python chapter_05_rag_agent.py`：

```
--- 开始构建向量数据库 ---
--- 向量数据库构建完成 ---

--- RAG Agent 已准备就绪，请输入您的问题 --- (输入'exit'退出)
> LangGraph是什么？
AI: LangGraph是一个由LangChain团队开发的库，用于构建有状态、多角色应用。它的核心思想是将Agent的执行流程表示为一个图（Graph）。

> 它是如何实现记忆的？
AI: LangGraph通过Checkpoint机制实现持久化记忆。

> 这本书的作者是谁？
AI: 上下文中没有提供关于这本书作者的信息。
```

**解读**：

-   当我们的问题（如“LangGraph是什么？”）与`knowledge.txt`中的内容相关时，`retriever`成功地“选择”了相关的文本块，并将其填充到Prompt的`{context}`中，使得LLM能够给出精准的回答。
-   当我们问一个文档中不存在的问题时（如“这本书的作者是谁？”），由于`retriever`没有找到相关上下文，LLM很诚实地回答它不知道。
-   这个过程完美地展示了**上下文治理**：我们没有将整个文档都塞给LLM，而是只提供了最相关的那一小部分，实现了高效、精准、低成本的问答。

---

## 5.4 本章小结

-   本章我们探讨了**上下文工程**的重要性，它是构建高效、经济、可控Agent的关键。
-   我们学习了**RAG（检索增强生成）**框架背后的四大核心操作：
    -   **Write**: 构建外部知识库。
    -   **Select**: 根据问题动态检索相关信息。
    -   **Compress**: 精简上下文，降本增效。
    -   **Isolate**: 保证不同任务上下文的独立性。
-   通过一个实战Lab，我们亲手构建了一个具备RAG能力的问答Agent，它能从本地文档中寻找答案，而不是仅仅依赖LLM的内部知识。
-   我们理解了RAG如何成为**上下文治理**的基石，它让Agent的知识边界变得可控、可扩展。

到目前为止，我们已经为Agent构建了行动、记忆和获取知识的能力。在下一部分，我们将进入更高级的主题：如何让Agent拥有更强的自主性，像人类一样进行规划、分解任务，并与其他Agent进行协作。我们将正式开始探索**代理性**的深层奥秘。
