# MCP协议架构详解

> 基于 [MCP官方规范](https://modelcontextprotocol.io) (2025-06-18版本) 编写

## 1. 什么是MCP

Model Context Protocol (MCP) 是由 Anthropic 发起的开放协议，旨在标准化 AI 应用与外部工具/数据源之间的上下文交换方式。它的核心理念是：**像 USB-C 统一了充电接口一样，MCP 统一了 AI Agent 与工具之间的连接标准。**

### 1.1 MCP解决的痛点

在 MCP 出现之前，每个 AI 应用要对接 N 个工具/数据源，需要写 N 套定制化的集成代码（M×N 问题）。MCP 通过标准化协议将其降为 M+N：

| 没有MCP | 有MCP |
|---------|-------|
| 每个AI应用为每个工具写适配器 | AI应用只需实现一个MCP Client |
| 每个工具为每个AI应用写接口 | 工具只需实现一个MCP Server |
| 集成成本：O(M×N) | 集成成本：O(M+N) |

### 1.2 MCP的范围边界

**MCP做的事：**
- 定义客户端与服务端之间的通信协议
- 规范工具发现、调用、结果返回的标准格式
- 提供资源（Resources）、提示词（Prompts）等上下文共享原语

**MCP不做的事：**
- 不规定 AI 应用如何使用 LLM
- 不管理 AI 应用的内部逻辑
- 不处理模型推理、训练等底层问题

---

## 2. 核心架构

### 2.1 参与者（Participants）

MCP 遵循 **客户端-服务端** 架构，包含三个关键角色：

```
┌─────────────────────────────────────────────────────┐
│                  MCP Host (AI Application)           │
│  例如: Claude Desktop / VS Code / 你的Agent应用       │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │MCP Client│  │MCP Client│  │MCP Client│  ...      │
│  │    #1    │  │    #2    │  │    #3    │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
└───────┼─────────────┼─────────────┼─────────────────┘
        │             │             │
   专用连接        专用连接        专用连接
        │             │             │
   ┌────▼─────┐  ┌────▼─────┐  ┌───▼──────┐
   │MCP Server│  │MCP Server│  │MCP Server│
   │(本地)    │  │(本地)    │  │(远端)    │
   │Filesystem│  │Database  │  │Sentry    │
   └──────────┘  └──────────┘  └──────────┘
```

**关键概念辨析：**

| 角色 | 定义 | 示例 |
|------|------|------|
| **MCP Host** | 协调和管理多个MCP Client的AI应用 | Claude Desktop, VS Code |
| **MCP Client** | 维护与单个MCP Server连接的组件 | Host内部的连接管理对象 |
| **MCP Server** | 向MCP Client提供上下文的程序 | 文件系统服务、数据库服务、API服务 |

> ⚠️ **面试要点**：MCP Server 指的是"提供上下文数据的程序"，不管它运行在本地还是远端。不要因为名字里有"Server"就认为它一定是远程的。

### 2.2 两层架构

MCP 由两个层次组成：

```
┌──────────────────────────────┐
│         Data Layer (内层)     │
│  JSON-RPC 2.0 协议           │
│  生命周期管理 + 核心原语       │
├──────────────────────────────┤
│       Transport Layer (外层)  │
│  Stdio / Streamable HTTP    │
│  连接建立 + 消息帧 + 认证     │
└──────────────────────────────┘
```

#### Data Layer（数据层）

数据层是 MCP 的核心，基于 **JSON-RPC 2.0** 协议，定义了：

1. **生命周期管理**：连接初始化、能力协商、连接终止
2. **Server Primitives**：Server 可以向 Client 提供的能力
   - **Tools**：可执行的操作（文件操作、API调用、数据库查询）
   - **Resources**：上下文数据源（文件内容、数据库记录、API响应）
   - **Prompts**：可复用的交互模板（系统提示词、few-shot示例）
3. **Client Primitives**：Client 可以向 Server 提供的能力
   - **Sampling**：允许Server请求Host的LLM生成补全
   - **Elicitation**：允许Server向用户请求更多信息
   - **Logging**：允许Server向Client发送日志
4. **Utility Primitives**：跨切面功能
   - **Notifications**：实时通知
   - **Progress**：长时间操作的进度追踪
   - **Tasks**（实验性）：持久化执行包装器

#### Transport Layer（传输层）

传输层管理通信通道和认证：

| 传输方式 | 适用场景 | 特点 |
|---------|---------|------|
| **Stdio** | 本地进程通信 | 使用标准输入/输出流，零网络开销，性能最优 |
| **Streamable HTTP** | 远端通信 | HTTP POST + 可选SSE流，支持标准HTTP认证 |

> 传输层的关键设计：它将通信细节从协议层抽象出来，使得**同样的 JSON-RPC 2.0 消息格式可以在所有传输机制上工作**。

---

## 3. 生命周期管理

MCP 是一个**有状态协议**，连接必须经过完整的生命周期：

```
Client                          Server
  │                                │
  │──── initialize (Request) ─────▶│  1. 协商版本和能力
  │◀─── initialize (Response) ─────│
  │                                │
  │─── initialized (Notification)─▶│  2. 确认就绪
  │                                │
  │◀═══ 正常通信阶段 ══════════════▶│  3. 工具发现/调用/通知
  │                                │
  │──── shutdown (Request) ───────▶│  4. 关闭连接
  │◀─── shutdown (Response) ───────│
  │                                │
```

### 3.1 初始化握手（Initialization）

初始化是 MCP 最关键的环节，完成三件事：

**1) 协议版本协商**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": { "elicitation": {} },
    "clientInfo": { "name": "my-agent", "version": "1.0.0" }
  }
}
```

**2) 能力发现**

Server 响应中声明自己支持的能力：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "tools": { "listChanged": true },
      "resources": {}
    },
    "serverInfo": { "name": "weather-server", "version": "2.0.0" }
  }
}
```

**3) 确认就绪**
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

> ⚠️ **面试高频考点**：版本协商（Version Negotiation）和能力协商（Capabilities Negotiation）是 MCP 标准协议定义的两个关键机制。
> - 版本协商：双方交换支持的协议版本列表，选定双方都支持的版本，如果没有交集则连接优雅失败
> - 能力协商：握手时双方声明支持的功能，Client 可以根据 Server 返回的能力决定启用哪些功能

---

## 4. 核心原语（Primitives）

### 4.1 Tools（工具）

Tools 是 MCP 中最常用的原语，代表**可执行的操作**。

**工具发现：**
```json
// Request
{ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }

// Response
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "weather_current",
        "title": "获取当前天气",
        "description": "查询指定城市的当前天气信息",
        "inputSchema": {
          "type": "object",
          "properties": {
            "location": { "type": "string", "description": "城市名称" },
            "units": { "type": "string", "enum": ["metric", "imperial"] }
          },
          "required": ["location"]
        }
      }
    ]
  }
}
```

**工具调用：**
```json
// Request
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "weather_current",
    "arguments": { "location": "北京", "units": "metric" }
  }
}

// Response
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      { "type": "text", "text": "北京当前温度25°C，晴，湿度45%" }
    ]
  }
}
```

### 4.2 Resources（资源）

Resources 提供**上下文数据**，是只读的数据源：
- 文件内容
- 数据库 Schema
- API 响应缓存
- 配置信息

### 4.3 Prompts（提示词模板）

Prompts 是**可复用的交互模板**：
- 系统提示词
- Few-shot 示例
- 特定场景的提示词模板

### 4.4 Notifications（通知）

通知是**无需响应的 JSON-RPC 消息**，用于实时状态同步：

```json
// Server通知Client工具列表发生变化
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed"
}
```

收到通知后，Client 通常会重新请求 `tools/list` 来刷新工具列表。

> ⚠️ **设计要点**：通知只在初始化时声明了 `"listChanged": true` 能力的 Server 上发送。这是能力协商的实际应用场景。

---

## 5. AI应用中的MCP工作流

以下是 AI 应用中使用 MCP 的典型伪代码：

```python
# 1. 初始化连接
async with stdio_client(server_config) as (read, write):
    async with ClientSession(read, write) as session:
        init_response = await session.initialize()
        
        # 2. 检查Server能力
        if init_response.capabilities.tools:
            app.register_mcp_server(session, supports_tools=True)
        
        # 3. 发现工具
        tools_response = await session.list_tools()
        available_tools = tools_response.tools
        
        # 4. 注册到LLM的工具列表
        conversation.register_available_tools(available_tools)
        
        # 5. 当LLM决定调用工具时
        async def handle_tool_call(tool_name, arguments):
            result = await session.call_tool(tool_name, arguments)
            conversation.add_tool_result(result.content)
        
        # 6. 处理通知
        async def handle_tools_changed():
            tools_response = await session.list_tools()
            app.update_available_tools(tools_response.tools)
            if conversation.is_active():
                conversation.notify_llm_of_new_capabilities()
```

---

## 6. 工业级实践要点

### 6.1 工具数量控制

根据 OpenAI 官方建议和 UC Berkeley 2025年论文《Measuring Agents in Production》：

| 建议 | 数值 | 原因 |
|------|------|------|
| 单次调用工具上限 | ≤20个 | 超过后模型选择准确率显著下降 |
| 理想工具数 | ≤10个 | 模型能稳定选择正确工具 |
| 生产Agent步骤限制 | ≤10步 | 83.3%的生产级Agent采用此策略 |

**工程实践**：使用分阶段工具注入（Phase-based Tool Injection），每个阶段只注入该阶段需要的工具子集。

### 6.2 错误处理

```python
# MCP工具调用的错误处理模式
try:
    result = await session.call_tool(tool_name, arguments)
    if result.isError:
        # 工具执行失败，返回错误信息给LLM让它调整策略
        return {"error": True, "message": result.content[0].text}
    return {"error": False, "data": result.content}
except MCPConnectionError:
    # 连接断开，尝试重连
    await session.reconnect()
except MCPTimeoutError:
    # 超时处理
    return {"error": True, "message": "Tool execution timed out"}
```

### 6.3 安全性考虑

1. **工具白名单**：每个Agent只给予必要的工具，防止越权
2. **参数校验**：基于 inputSchema 做严格的参数验证
3. **沙箱隔离**：每个MCP Server在独立进程中运行
4. **认证**：远程Server使用OAuth获取认证token

---

## 7. 常见面试问题

### Q1: MCP协议和传统的REST API有什么区别？

| 维度 | MCP | REST API |
|------|-----|----------|
| 设计目标 | AI Agent与工具交互 | 通用Web服务间通信 |
| 通信模式 | 双向（Client和Server都可以发请求） | 请求-响应（Client发，Server回） |
| 工具发现 | 内置tools/list动态发现 | 需要OpenAPI/Swagger文档 |
| 状态管理 | 有状态协议，需要生命周期管理 | 通常无状态 |
| 能力协商 | 内置版本和能力协商 | 无标准机制 |
| 实时通知 | 内置notification机制 | 需要WebSocket等额外机制 |

### Q2: 如果MCP Client和Server版本不兼容怎么办？

MCP 标准协议定义了两个关键机制：

1. **版本协商**：初始化时双方交换支持的协议版本，没有交集则连接优雅失败
2. **能力协商**：握手时声明支持的功能，Client根据Server能力决定是否启用特定功能

客户端在发现不兼容时可以：
- 回退到兼容版本（如果支持多版本）
- 禁用不支持的功能（避免调用导致错误）
- 提示用户"当前服务端不兼容，需要升级或切换"

### Q3: MCP Server的工具列表可以动态变化吗？

可以。MCP 支持 `notifications/tools/list_changed` 通知机制：
1. Server 在初始化时声明 `"listChanged": true` 能力
2. 当工具列表发生变化时，Server 发送通知
3. Client 收到通知后重新请求 `tools/list`
4. AI应用更新LLM的可用工具列表

这在动态环境中非常重要：工具可能根据服务器状态、外部依赖或用户权限而变化。

---

## 8. 参考资料

- [MCP官方文档](https://modelcontextprotocol.io/docs/learn/architecture)
- [MCP规范仓库](https://github.com/modelcontextprotocol/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- 协议版本：2025-06-18（最新）
