@echo off
REM Tier-1 deterministic guardian for the Homerun trading harness.
REM Enforces risk limits (daily loss, gross exposure, open orders, heartbeat,
REM and the account-equity catastrophe floor). On breach it kill-switches,
REM emergency-stops live, stops live, and blocks every trader. No LLM involved.
REM Scheduled every few minutes by Windows Task Scheduler.
cd /d C:\homerun
C:\Python314\python.exe scripts\monitoring\homerun_caretaker.py guard ^
  --policy scripts\monitoring\homerun_caretaker_live_policy.json ^
  --report-dir output\caretaker >> output\caretaker\guard_cron.log 2>&1
