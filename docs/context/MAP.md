# HOMERUN 代码地图

核对基线：2026-08-11，Git `389d24699479daf102bfa9c22882375ca9ebb07a`。本文件只描述入口、责任和边界；不复制实现。未逐项运行验证的内容标为「未运行核验」。

## 目录与主要模块

- `.github/`：上游 GitHub 元数据和 CI；实现方不得修改，不放业务配置。
- `.greencheck/`、`.greencheck.yml`：仓库自带的检查配置；不放运行时状态。
- `backend/`：FastAPI、worker、策略、执行、数据和持久化后端；不放前端组件。
- `backend/api/`：HTTP/WebSocket 路由和 DTO 边界；不承载长循环、策略判断或直接数据库迁移。
- `backend/main.py`：API 进程生命周期、路由注册和健康检查；不应成为业务策略容器。
- `backend/config.py`：环境默认值与运行时设置映射；密钥值不得写入源码或文档。
- `backend/models/`：Pydantic 领域对象、SQLAlchemy 模型和 session 工厂；不放 provider 网络调用。
- `backend/models/opportunity.py`：`Opportunity` 与 `ExecutionPlan` 契约；策略和执行链共享，不放钱包采集逻辑。
- `backend/models/database.py`：数据库模型、连接池、初始化和事务基础设施；文件已很大，不应继续塞入策略计算。
- `backend/alembic/`：PostgreSQL schema 迁移；任何新迁移都必须先有 ADR。
- `backend/interfaces/`：跨模块接口定义；不放具体 provider 实现。
- `backend/services/`：可复用业务服务、provider client、缓存、事件和执行协调；不放 API 路由。
- `backend/services/strategies/`：独立检测/评估/退出策略插件；不得自行改变 `Trader.mode`、数据库 schema 或直接绕过 orchestrator 下单。
- `backend/services/opportunity_strategy_catalog.py`：系统策略种子与动态策略目录；不应覆盖用户编辑后的策略配置。
- `backend/services/data_events.py`：`DataEvent`/`EventType` 契约；新增事件类型属于接口变化，必须有 ADR。
- `backend/services/strategy_signal_bridge.py`：把 `Opportunity` 发布到 intent runtime；不负责重新计算策略收益。
- `backend/services/trader_orchestrator/`：信号 gate、风险、execution plan、shadow/live 提交、退出和核账；策略不得绕过此边界。
- `backend/services/recorded_event_bus/`：topic catalog、记录、Parquet/SQL 适配与 replay；不把未确认事件冒充最终成交事实。
- `backend/services/wallet_ws_monitor.py`：现有钱包交易事件、持久化和 callback；SRH 设计把其旧行为列为兼容性边界。
- `backend/services/wallet_rtds_feed.py`：现有钱包 RTDS feed；不承载 SRH 共识判断。
- `backend/services/polymarket.py`：Polymarket HTTP 数据访问与缓存；不放策略门槛。
- `backend/workers/`：按周期或事件运行的后台 worker；每个 worker 经 `host.py` 注册和独立监控。
- `backend/workers/host.py`：trading/news/discovery/jobs/recording/detection/reconciliation/services plane 的进程内编排；不放具体策略。
- `backend/workers/tracked_traders_worker.py`：现有钱包 rollup、confluence 和 `TRADER_ACTIVITY` 发布；SRH 不得静默改变其 payload 或清扫语义。
- `backend/tests/`：既有可执行规格；实现方不得修改测试文件，测试争议写入 PR。
- `backend/scripts/`：离线校准、导入和运维脚本；不得被运行时隐式调用形成第二条资金路径。
- `backend/utils/`：时间、日志、重试、密钥等通用工具；不放领域策略。
- `backend/data/`：随源码分发的 JSON 基线数据；不放本地账户、密钥或运行数据库。
- `frontend/`：React/TypeScript/Vite 客户端；不持有真实交易密钥。
- `frontend/src/components/`：界面和交互组件；不复制后端策略或 PnL 计算。
- `frontend/src/components/ui/`：通用 UI 基础组件；不含领域数据访问。
- `frontend/src/services/`：Axios API client 和响应类型；后端契约变化需同步，但不得自行定义另一套真相。
- `frontend/src/hooks/`：WebSocket、刷新和交互 hooks；不直接提交交易。
- `frontend/src/store/`：Jotai 客户端状态；不是持久账本。
- `frontend/src/lib/`：前端纯工具、时间和展示转换；不计算资金真相。
- `frontend/src/i18n/`：多语言资源；不放运行配置。
- `scripts/infra/`：本地安装、启动、健康检查和 Docker 辅助；固定源码二开应走本地构建，不依赖浮动 `latest`。
- `scripts/launchers/`：桌面启动入口；不放业务逻辑。
- `scripts/ci/`：本地/CI 检查脚本；实现方不得修改 `.github/workflows/`。
- `scripts/maintenance/`：显式运行的修复与核账脚本；执行前必须核对目标和回滚。
- `scripts/ml/`：离线模型构建；输出不得自动晋级实盘。
- `scripts/monitoring/`：运行监控和 caretaker；不替代交易账本。
- `tools/`：延迟、性能和日志辅助工具；不进入正常资金路径。
- `docs/`：用户文档、历史 handoff、ADR 和项目记忆；历史 handoff 不自动等于现行架构约束。
- `docs/context/`：供实现与独立评审共享的项目记忆；每个 PR 必须同步更新。
- `screenshots/`：README/UI 图片；不作为功能或收益证据。
- `docker-compose.yml`：PostgreSQL、Redis、API、部分 worker plane 和前端的默认编排；当前默认 compose 未显式启动 host 中全部 plane。
- `gui.py`：桌面启动 GUI；不承载后端策略。

## 数据流

1. Polymarket/Kalshi/Binance、新闻、天气等外部数据经 provider client、WebSocket feed 和 worker 进入缓存、数据库、recorded-event topic 或 `DataEvent`。
2. 动态策略按 `source_key`/subscription 消费市场或事件数据，输出 `Opportunity`；策略不决定 shadow/live，也不直接调用 CLOB。
3. `strategy_signal_bridge` 把机会交给 intent runtime/`TradeSignal`，trader orchestrator 依次执行策略 evaluate、平台/账户风险 gate、仓位和 `ExecutionPlan`。
4. `Trader.mode` 决定订单进入 shadow 模拟还是 live provider；结果写入 `TraderOrder`、position、verification 和事件账本，再由 API/WebSocket 提供给前端。
5. 多进程 plane 通过 PostgreSQL 与 Redis 协作；数据库和 verification 是资金真相，UI 快照、README 宣称和 UNLOGGED 聚合不是收益真相。

## Python 直接依赖

- `fastapi`：提供 HTTP API、依赖注入和路由生命周期。
- `uvicorn[standard]`：运行 FastAPI ASGI 服务。
- `mcp`：把回测/迭代工具暴露给外部 agent；是否启用未运行核验。
- `sse-starlette`：为 MCP/流式响应提供 SSE。
- `python-multipart`：解析表单和文件上传。
- `httpx[socks]`：异步 HTTP provider client，并支持 SOCKS 代理。
- `websockets`：连接市场、钱包和 provider WebSocket。
- `pydantic`：API 与领域数据校验。
- `pydantic-settings`：环境变量和设置加载。
- `jsonschema`：动态策略/数据源 schema 校验。
- `sqlalchemy[asyncio]`：异步 ORM、事务和连接池。
- `alembic`：数据库 schema 版本迁移。
- `asyncpg`：PostgreSQL 异步驱动。
- `greenlet`：SQLAlchemy 异步桥接运行依赖。
- `uvloop`：Linux/macOS worker 快事件循环；可选、按平台安装。
- `winloop`：Windows worker 快事件循环；可选、按平台安装。
- `cryptography`：数据库内密钥的 Fernet 加解密。
- `pyarrow`：Parquet 数据集、recorded-event 和回测数据读取。
- `redis`：跨进程 bus、状态传播和短期缓存。
- `urllib3`：约束 HTTP 传递依赖兼容版本。
- `pyyaml`：读取 YAML 配置/策略材料。
- `markdown`：将 Markdown 转成报告或界面内容。
- `feedparser`：RSS/Atom 新闻源解析。
- `numpy`：数值、向量和模型输入计算。
- `scikit-learn`：传统 ML、校准和评估。
- `onnxruntime`：运行 ONNX 模型。
- `xgboost-cpu`：CPU 梯度提升模型。
- `lightgbm`：梯度提升模型训练/推断。
- `sentence-transformers`：新闻和市场语义嵌入。
- `faiss-cpu`：语义向量索引与近邻检索。
- `lifelines`：shadow fill 的 Cox 生存模型。
- `weasyprint`：从 HTML 生成策略/研究 PDF。
- `jinja2`：报告模板渲染。
- `pytest-asyncio`：异步 pytest 测试支持。
- `pytest-timeout`：阻止测试无限挂起。
- `py-clob-client-v2`：Polymarket live CLOB 请求与订单接口；仅实盘额外安装。
- `eth-account`：EIP-712/以太坊账户签名；仅实盘需要。
- `web3`：Polygon/链上 RPC 交互；仅实盘需要。
- `coincurve`：C-backed secp256k1 签名，降低交易 loop 的 GIL/CPU 成本。

## 前端与工具直接依赖

- `react`：前端组件运行时；manifest 当前为 `^19.2.4`。
- `react-dom`：React 浏览器渲染；manifest 当前为 `^19.2.4`。
- `@tanstack/react-query`：API 查询、缓存和失效。
- `@tanstack/react-virtual`：大列表虚拟渲染。
- `axios`：统一 HTTP API client。
- `jotai`：客户端全局和持久状态。
- `i18next`：多语言资源管理。
- `i18next-browser-languagedetector`：浏览器语言识别。
- `react-i18next`：React 的 i18n 绑定。
- `@assistant-ui/react`：AI 对话界面组件。
- `@assistant-ui/react-markdown`：AI 对话 Markdown 渲染。
- `@openuidev/react-lang`：OpenUI 语言/渲染支持；具体使用边界未逐项审计。
- `@openuidev/react-ui`：OpenUI React 组件；具体使用边界未逐项审计。
- `@codemirror/autocomplete`：策略/数据源编辑器自动补全。
- `@codemirror/commands`：CodeMirror 编辑命令。
- `@codemirror/lang-json`：JSON 编辑语法。
- `@codemirror/lang-python`：Python 策略编辑语法。
- `@codemirror/language`：CodeMirror language 基础设施。
- `@codemirror/lint`：编辑器 lint 标记。
- `@codemirror/search`：编辑器搜索。
- `@codemirror/state`：CodeMirror 状态模型。
- `@codemirror/theme-one-dark`：编辑器暗色主题。
- `@codemirror/view`：CodeMirror 浏览器视图。
- `@radix-ui/react-collapsible`：可折叠 UI 原语。
- `@radix-ui/react-dialog`：对话框 UI 原语。
- `@radix-ui/react-dropdown-menu`：下拉菜单 UI 原语。
- `@radix-ui/react-label`：表单标签 UI 原语。
- `@radix-ui/react-popover`：浮层 UI 原语。
- `@radix-ui/react-progress`：进度条 UI 原语。
- `@radix-ui/react-scroll-area`：滚动区域 UI 原语。
- `@radix-ui/react-select`：选择框 UI 原语。
- `@radix-ui/react-separator`：分隔线 UI 原语。
- `@radix-ui/react-slot`：组合式组件 slot。
- `@radix-ui/react-switch`：开关 UI 原语。
- `@radix-ui/react-tabs`：标签页 UI 原语。
- `@radix-ui/react-toggle`：切换按钮 UI 原语。
- `@radix-ui/react-toggle-group`：成组切换 UI 原语。
- `@radix-ui/react-tooltip`：提示 UI 原语。
- `class-variance-authority`：组件 variant class 组合。
- `clsx`：条件 class 拼接。
- `cmdk`：命令面板交互。
- `framer-motion`：界面动画。
- `lightweight-charts`：价格/时间序列图表。
- `liveline`：实时线图展示；具体使用边界未逐项审计。
- `lucide-react`：界面图标。
- `maplibre-gl`：地图渲染。
- `react-is`：React 元素类型判断；具体直接使用未逐项审计。
- `react-syntax-highlighter`：代码高亮。
- `recharts`：统计与仪表盘图表。
- `tailwind-merge`：合并 Tailwind class 冲突。
- `tailwindcss-animate`：Tailwind 动画工具类。
- `@types/react`、`@types/react-dom`：TypeScript React 类型声明。
- `@vitejs/plugin-react`：Vite React 编译插件。
- `autoprefixer`：CSS 浏览器前缀。
- `postcss`：Tailwind/CSS 构建管线。
- `tailwindcss`：实用类 CSS 构建。
- `typescript`：前端静态类型和编译。
- `vite`：前端开发与生产构建。
- `vite-node`：Node 环境执行 Vite 模块；具体使用边界未逐项审计。
- `tree-sitter`：launcher tooling 的语法树基础。
- `tree-sitter-powershell`：PowerShell launcher 语法检查。

## 外部服务

- PostgreSQL：持久业务状态、策略、订单、仓位和 verification 的主数据库。
- Redis：plane 间信号、事件和状态传播；部分消费者可降级到内存，但行为需逐模块核验。
- Polymarket Gamma/Data/CLOB/RTDS：市场发现、公开钱包活动、盘口/成交和订单执行。
- Polygon RPC/WebSocket：链上状态、账户和结算相关读取。
- Kalshi HTTP/WebSocket：跨平台市场和交易数据。
- Binance WebSocket：加密现货基准价格。
- Chainlink：部分加密市场结算/参考价格。
- RSS/GDELT：新闻和事件采集。
- Open-Meteo 等天气源：天气预测输入；具体 adapter 以 `services/weather` 和测试为准。
- Telegram Bot API：可选通知通道。
- Hugging Face：首次拉取语义模型；token 可选。
- LLM provider：由数据库/UI provider 配置驱动；当前已配置哪家不在源码中确定。

## 项目术语

- `Opportunity`：策略检测出的候选机会，不等于已批准或已成交订单。
- `DataEvent`：触发动态策略的标准事件载体。
- `source_key`：策略所属数据/运行通道，如 scanner、traders、crypto、sports、news、weather。
- `worker plane`：隔离 event loop、CPU/内存和资金热路径的独立进程分组。
- `intent runtime`：把机会转成可供 trader 消费的运行时信号层。
- `ExecutionPlan`：通过 gate 后的具体交易腿、价格政策和执行约束。
- `shadow`：微观结构模拟，不向 venue 发送真钱订单；仍可能存在模型误差。
- `live`：经 provider/CLOB 发送真实订单。
- `verified PnL`：经过订单 verification 链核对的盈亏；未验证 `$0`、理论 payout 和 expected ROI 不属于它。
- `stable_id`：同一策略/市场候选跨扫描的稳定身份，用于去重而非订单幂等的全部真相。
- `TRADER_ACTIVITY`：现有 tracked-trader 数据事件；兼容性受保护。
- `SRH`：拟议 Smart-flow Retail Handoff 钱包共识策略，尚未实现，待 ADR-000 追认。
- `MIN`：拟议 Marginal Information Network 钱包依赖去重/权重内核，尚未实现。
- `CORE_MAKER`：拟议 SRH 被动挂单档位，不等于保证成交或盈利。
- `ENHANCED_TAKER`：拟议 SRH 强证据主动成交档位，必须扣动态费用、滑点和退出成本。
- `point-in-time`：只使用决策时刻已经可见的数据，禁止事后钱包排名、结算结果或未来盘口回填。
- `retail-like`：按当时可观察行为构造的匹配对照，不等于对真实个人身份的断言。
