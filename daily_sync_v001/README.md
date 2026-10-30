# Daily Git sync
Runs at 23:30 server local time (Asia/Shanghai).
Scope: /mnt/data2/wjy_SERL/Embodied_Safety only. Sibling franka_sim is NOT included.
Uses repository .gitignore, git add -A, dated commit if changed, git push origin main, and remote HEAD verification.
Never force-pushes or automatically merges. On failure, preserves the local commit and retries the next day.
Logs: outputs/daily_sync/YYYY-MM-DD.log (ignored).
Read-only check: python3 -B daily_sync_v001/daily_push.py --check
Cron entry is installed for ufdwaa. No immediate commit/push during installation.
