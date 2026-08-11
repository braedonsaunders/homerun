# 当前交接状态

基线日期：2026-08-11
源码：`389d24699479daf102bfa9c22882375ca9ebb07a`（浅克隆，仅 1 个可见 commit）
当前分支：`chore/bootstrap-project-context`

## 现在能跑什么

- 仓库已有完整 FastAPI、worker plane、React 前端、PostgreSQL/Redis、shadow/live orchestrator 和大量测试代码。
- 后端隔离 venv 已按现有 requirements 安装 158 个包，`python -m pip check` 返回 `No broken requirements found`。
- 前端已按现有 lockfile 执行 `npm ci`，`npm run build` 成功；构建产物不纳入本 PR。
- `.gitignore` 已补充环境文件、私钥和本地凭证规则；`.env.example` 继续受 Git 跟踪。
- 最终暂存后的 provider token、私钥、JWT、引号内凭证和敏感路径扫描，在实际 index、当前受跟踪树、可见 HEAD 与 commit message 中均为 0 命中；宽泛赋值规则的候选均为代码变量表达式。
- 钱包共识研究、正式设计和分阶段实施计划在仓库外的独立研究目录中，尚未进入业务源码。

## 现在还不能跑什么

- 没有 `docs/adr/ADR-000*`，也没有对应 GitHub issue；按新职责约定，SRH 业务实现必须停止。
- 没有 `wallet_srh_worker`、`WALLET_FLOW`、SRH 事实表或 `wallet_srh_v1` 策略；这些仍是待评审设计，不是现有功能。
- 后端单进程全量 pytest 不能全绿：使用独立临时 PostgreSQL 后，2092 个测试通过，随后 Windows 创建 asyncio socketpair 时发生 `WinError 10055`，测试在约 81% 停止。
- 从中断文件开始的新进程补跑 44 个剩余文件得到 518 passed、1 个同类 setup error；两个触发测试均在全新进程单独通过，未出现业务断言失败。
- 测试退出还报告未清理的 FeedManager cache eviction、recorded-event flusher 任务和未 await 的连接取消；这属于上游基线资源生命周期风险，尚未修复。
- 没有真实 forward shadow 订单和 verified PnL；当前不能声称钱包策略可赚钱。
- 没有完整 Git 历史的密钥审计：本机无 gitleaks/trufflehog/detect-secrets，且 clone 是 shallow。
- GitHub 用户 `65241946` 的 fork 已建立为本地 `fork` 远端；当前分支已推送，上游 PR 为 `braedonsaunders/homerun#317`。
- Docker 中存在旧 `homerun-pmr` 的停止 PostgreSQL 容器并绑定旧数据目录；本轮未复用或删除它。测试曾走 `127.0.0.1:55433` 的独立 tmpfs 临时数据库，验证后已停止并自动删除。

## 我正在做的

- 本 PR 只做项目接管基线：更新 `.gitignore`、替换根 `AGENTS.md`、建立四份 `docs/context/` 记忆。
- 没有修改后端、前端、测试、数据库、ADR 或 workflow。
- 分支无 issue 编号是已知规格缺口；因为本次任务发生在 issue/ADR 工作流建立之前，使用 `chore/bootstrap-project-context`。

## 下一步我打算做什么

1. 停止业务开发，等待 Claude 对 PR #317 做基线评审并提交 `ADR-000`。
2. 架构方创建带验收标准的 GitHub issue 后，才从新分支按 ADR、issue 和既有失败测试实现。
3. Windows pytest 资源生命周期、前端依赖漏洞等旁支问题分别建 issue，不混入首个 SRH 功能 PR。

## 已知的坑 —— 别踩

- Windows 文件系统大小写不敏感：原 `agents.md` 与目标 `AGENTS.md` 是同一路径，必须确认 Git 最终记录正确文件名。
- `origin` 是上游 `braedonsaunders/homerun`，`fork` 才是用户 `65241946/homerun`；bootstrap 分支只能推到 `fork`，PR base 才指向上游 main。
- 当前 clone 只有 1 个可见 commit；`git log`、secret scan 和历史判断都不代表完整上游历史。
- Windows 下长进程运行约 2000 个异步测试后会耗尽 socketpair 资源；触发用例单独通过，不能据此误改交易代码，也不能把基线写成全绿。
- pytest 退出日志显示后台 task 和 asyncpg 取消协程未完全回收；应由单独 issue/ADR 决定测试生命周期修复，不塞进 bootstrap PR。
- `npm audit` 报告 17 个既有漏洞（1 low、7 moderate、9 high），Vite 还报告超大 chunk；升级依赖或拆包属于独立工作。
- 旧 `docs/plane_isolation_handoff.md` 写着“local commits, not pushed”，但当前源码已有 reconciliation plane；它可能过时，不是 ADR。
- 旧 `agents.md` 写 React 18，当前 `frontend/package.json` 是 React/React DOM `^19.2.4`；版本信息必须以 manifest/lock 为准。
- Polymarket wallet trades 的 API 默认行为可能漏 maker；实现时必须重新核验 `takerOnly`，不能只信旧研究快照。
- 旧 `TRADER_ACTIVITY` 同时服务 confluence/copy；SRH 不得改变其 payload、旧 callback 数或信号清扫语义。
- `bridge_opportunities_to_signals` 的 `sweep_missing=True` 会影响同 source 的其他信号；独立 worker 设计必须有隔离测试。
- 通用 position lifecycle 当前没有跨进程钱包流输入；不得为 SRH 偷改全局退出契约。
- `backend/models/database.py` 很大且承载连接池/事务/模型；任何 schema 变更都先等 ADR，并用迁移而不是运行时补列。

## 技术债

- 数据库模型集中在单一大文件，变更冲突和认知成本高；当前不重构，因为这会扩大 bootstrap PR。
- README/旧 agent 文档与当前 manifest 存在版本漂移；本 PR 只在 context 中标记，不顺手改产品文档。
- 默认 compose 启动的 worker plane 少于 `host.py` 当前支持的 plane；是否是部署设计或遗漏尚不确定，交给 ADR-000 评审。
- SRH 设计需要新 schema 和事件契约，按新规则必须由架构方先批准，不能因已有实施计划而跳过 ADR。
- Windows 全量测试的后台任务/事件循环清理不完整；目前只能通过分进程补跑验证断言，尚无经过 ADR 的长期修复。
- 前端依赖审计和 bundle 体积存在已知风险；本 PR 不引入依赖升级或构建配置变化。
