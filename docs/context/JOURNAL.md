# 开发日志

## 2026-08-11 — 项目记忆与 Git 安全基线

- 在 `chore/bootstrap-project-context` 上补充密钥忽略规则，并建立 MAP、DECISIONS、HANDOFF、JOURNAL。
- index、当前树和浅克隆可见历史未命中 provider token、私钥、JWT、引号内凭证或敏感路径；本机没有专用 scanner 且历史仅 1 个 commit，结论必须限缩。
- 全局 Python 环境不匹配，改用隔离 venv；前端 lockfile 构建通过，但 npm 审计有 17 个既有漏洞。
- 后端单进程在 2092 passed 后因 Windows socket 资源耗尽停止；剩余文件补跑 518 passed/1 个同类 setup error，两个触发用例单独均通过，且退出暴露未清理后台任务。
- 原 `agents.md` 已按用户要求替换；没有修改业务代码、测试、ADR 或 workflow。bootstrap 已提交为 PR #317，推翻“可直接进入 SRH 实现”的先前安排，现停下等 `ADR-000`。

## 2026-08-11 — 钱包共识重新研究与实施计划

- 固定 HOMERUN SHA，从官方接口、真实源码和社区项目重新审计钱包链，而不是继承旧本地改动。
- 发现 maker 历史缺口、旧 rollup 身份损失和收益核账边界；设计收紧为独立事实/worker/event/strategy 旁路。
- SRH+MIN 仍是假设，不是已证实收益；14 个工程检查项归并为数据旁路、策略闭环、真实模拟与安全验收三个阶段。
- 该阶段只生成研究、正式设计和实施计划，没有修改 HOMERUN 业务代码。
