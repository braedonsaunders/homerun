@echo off
REM Tier-2 deterministic operator for the Homerun trading harness.
REM Multi-day cumulative-bleed kill on the managed canary (per-trader limits
REM only catch single-day loss) + realized-PnL hygiene + safety enforcement.
REM No LLM. Scheduled every 15 min by Windows Task Scheduler.
cd /d C:\homerun
C:\Python314\python.exe scripts\monitoring\homerun_caretaker.py operator ^
  --policy scripts\monitoring\homerun_caretaker_live_policy.json ^
  --report-dir output\caretaker >> output\caretaker\operator_cron.log 2>&1
