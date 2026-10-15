#!/usr/bin/env bash
# watch_stalls.sh: sample everything needed to attribute a vLLM stall on the dual-Spark cluster.
#   scripts/watch_stalls.sh                 # 2 s sampling until Ctrl-C
#   scripts/watch_stalls.sh --once          # one snapshot (cron / quick look)
#   INTERVAL=5 OUT=results/stall-watch/x.csv scripts/watch_stalls.sh
#
# Columns exist to separate the candidate causes of "high load, no tokens", no post-hoc guessing:
#   run / wait / wait_cap / wait_deferred / kv
#                       scheduler state. wait_cap>0 = not enough blocks; wait_deferred>0 = the scheduler is
#                       holding back admitted work; both 0 with wait>0 = something else is blocking
#   gen_delta           engine progress since the last sample: 0 while run>0 IS the stall
#   gpu_util            GPU busy while gen_delta=0 => the engine is waiting, not computing (GB10 reports no
#                       gpu memory figure: memory.used is [N/A])
#   avail / swap_kb_s / majfault / psi_mem  host memory pressure. On a GB10 (unified memory) the engine's
#                       host pages are reclaimable: eviction turns the next step that touches them into
#                       NVMe page-faults, i.e. a stall that looks like load with no output.
#   roce_kb_s           rank-to-rank traffic: if this goes flat, the TP collective went quiet
#   w_avail / w_majfault   the same memory story on the worker
# Writes CSV to $OUT (default results/stall-watch/<timestamp>.csv). Read-only; changes nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f cluster.env ] && . ./cluster.env
API=${API:-http://localhost:8000}
IBDEV=${IBDEV:-rocep1s0f0}
INTERVAL=${INTERVAL:-2}
OUT=${OUT:-results/stall-watch/$(date +%Y%m%d-%H%M%S).csv}
mkdir -p "$(dirname "$OUT")"
HDR='ts,run,wait,wait_cap,wait_deferred,kv,gen_delta,gpu_util,mem_avail_gib,swapout_kb_s,majfault_delta,psi_mem,roce_kb_s,w_avail_gib,w_majfault_delta'
[ -f "$OUT" ] || echo "$HDR" | tee "$OUT"
[ "${1:-}" = --once ] && INTERVAL=0

on() { ssh -o BatchMode=yes "$1" "$2"; }
snap() {
  on "$HEAD" "
    curl -s -m 3 $API/metrics | awk '
      /^vllm:num_requests_running\{/{r=\$2} /^vllm:num_requests_waiting\{/{w=\$2}
      /^vllm:kv_cache_usage_perc/{k=\$2} /^vllm:gpu_cache_usage_perc/{k=\$2}
      /^vllm:generation_tokens_total\{/{g+=\$2}
      END{printf \"run=%s wait=%s waitcap=%s waitdef=%s kv=%s gen=%s\", r, w, wc, wd, k, g}'
    awk '/MemAvailable/{printf \" avail=%s\", sprintf(\"%.2f\", \$2/1048576)}' /proc/meminfo
    awk '/^pswpin|^pswpout|^pgmajfault/{gsub(\"pswp\",\"swap\");printf \" %s=%s\", \$1, \$2}' /proc/vmstat
    awk '{split(\$2,a,\"=\");printf \" psi=%s\", a[2]; exit}' /proc/pressure/memory
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr , ' ' | awk '{printf \" gpu_util=%s\", \$1}'
    printf ' roce=%s\n' \$(cat /sys/class/infiniband/$IBDEV/ports/1/counters/port_xmit_data)"
  on "$WORKER" "awk '/MemAvailable/{printf \" w_avail=%s\", sprintf(\"%.2f\", \$2/1048576)}' /proc/meminfo
                 awk '/^pgmajfault/{printf \" w_majfault=%s\", \$2}' /proc/vmstat"
}
row() { python3 -c '
import sys, time
p, c, dt = dict(l.split("=") for l in sys.argv[1].split()), dict(l.split("=") for l in sys.argv[2].split()), float(sys.argv[3])
import re
f = lambda d, k: float((re.findall(r"[-0-9.]+", d.get(k, "0")) or ["0"])[0])
print(",".join(str(x) for x in [time.strftime("%FT%TZ", time.gmtime()), f(c,"run"), f(c,"wait"), f(c,"waitcap"), f(c,"waitdef"), f(c,"kv"),
      f(c,"gen")-f(p,"gen"), f(c,"gpu_util"), f(c,"avail"),
      (f(c,"pswpout")-f(p,"pswpout"))*4/1024/dt, f(c,"pgmajfault")-f(p,"pgmajfault"), f(c,"psi"),
      (f(c,"roce")-f(p,"roce"))*4/1024/dt, f(c,"w_avail"), f(c,"w_majfault")-f(p,"w_majfault")]))' "$1" "$2" "$3"; }

prev=$(snap); t0=$(date +%s)
while :; do
  sleep "$INTERVAL"
  cur=$(snap); now=$(date +%s); line=$(row "$prev" "$cur" "$((now - t0))")
  echo "$line" | tee -a "$OUT"
  prev=$cur; t0=$now; [ "$INTERVAL" = 0 ] && break
done
