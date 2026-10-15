#!/usr/bin/env bash
# watch_stalls.sh: sample everything needed to attribute a vLLM stall on the dual-Spark cluster.
#   scripts/watch_stalls.sh                 # 2 s sampling until Ctrl-C
#   scripts/watch_stalls.sh --once          # one snapshot (cron / quick look)
#   INTERVAL=5 OUT=results/stall-watch/x.csv scripts/watch_stalls.sh
#
# Columns exist to separate the candidate causes of "high load, no tokens", without post-hoc guessing:
#   run / wait / wait_cap / wait_deferred
#                       scheduler state. wait_cap>0 = not enough KV blocks; wait_deferred>0 = the scheduler is
#                       holding back work it already admitted; wait>0 with both 0 = blocked elsewhere
#   kv                  KV pool usage. Creeping while gen_delta ~0 == the engine is *prefilling*, not stalled
#   gen_delta           decode tokens produced in the interval; near-zero with run>0 is the stall itself
#   prefill_recomputed  prompt tokens recomputed in the interval (the cost of a prefix-cache miss)
#   prefix_cache_hit_alloc  prompt tokens served from cache in the interval. KV creeping with gen_delta~0,
#                       prefill_recomputed~0 and this rising = big cached prompts being *admitted*, not recomputed
#   prefix_hit_pct      window-local prefix-cache hit rate. Low + high prefill_recomputed = someone's prompt
#                       diverged from its cached prefix (or its blocks were evicted): a cold prefill
#   gpu_util            busy while gen_delta=0 => the engine waits rather than computes. GB10 reports no GPU
#                       memory figure (memory.used is [N/A])
#   avail / swapout_kb_s / majfault_delta / psi_mem
#                       host memory pressure. On unified memory the engine's host pages are reclaimable, so
#                       eviction turns the next touch into NVMe page-faults: load with no output.
#                       psi_mem (PSI "some") is the honest signal; Majflt counts show only when the
#                       pages being faulted back in are swapped-out ones.
#   roce_kb_s           rank-to-rank traffic: flat + gen_delta 0 means the TP collective went quiet;
#                       high (1+ GB/s) + gen_delta ~0 means prefill is eating the step budget
#   w_avail / w_majfault  the same memory story on the worker
# Writes CSV to $OUT (default results/stall-watch/<timestamp>.csv). Read-only; changes nothing on the nodes.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f cluster.env ] && . ./cluster.env
API=${API:-http://localhost:8000}
IBDEV=${IBDEV:-rocep1s0f0}
INTERVAL=${INTERVAL:-2}
OUT=${OUT:-results/stall-watch/$(date +%Y%m%d-%H%M%S).csv}
mkdir -p "$(dirname "$OUT")"
HDR='ts,run,wait,wait_cap,wait_deferred,kv,gen_delta,prefill_recomputed,prefix_cache_hit_alloc,preemptions,prefix_hit_pct,gpu_util,mem_avail_gib,swapout_kb_s,majfault_delta,psi_mem,roce_kb_s,w_avail_gib,w_majfault_delta'
[ -f "$OUT" ] || echo "$HDR" | tee "$OUT"
[ "${1:-}" = --once ] && INTERVAL=0

AWK_METRICS='
{
  if (index($0,"vllm:num_requests_running{")==1)                 r=$2
  else if (index($0,"vllm:num_requests_waiting{")==1)            w=$2
  else if (index($0,"reason=\"capacity\"")>0 && /num_requests_waiting_by_reason/) wc=$2
  else if (index($0,"reason=\"deferred\"")>0 && /num_requests_waiting_by_reason/) wd=$2
  else if (index($0,"vllm:kv_cache_usage_perc")==1 || index($0,"vllm:gpu_cache_usage_perc")==1) k=$2
  else if (index($0,"vllm:generation_tokens_total{")==1)         g+=$2
  else if (index($0,"prompt_tokens_by_source_total")>0 && index($0,"local_compute")>0) comp=$2
  else if (index($0,"prompt_tokens_by_source_total")>0 && index($0,"local_cache_hit")>0) hit=$2
  else if (index($0,"vllm:num_preemptions_total{")==1) pre=$2
  else if (index($0,"vllm:prefix_cache_hits_total{")==1)         ph=$2
  else if (index($0,"vllm:prefix_cache_queries_total{")==1)     pq=$2
}
END{printf "run=%s wait=%s waitcap=%s waitdef=%s kv=%s gen=%s comp=%s hit=%s preempt=%s pq=%s ph=%s", r, w, wc, wd, k, g, comp, hit, pre, pq, ph}'

on() { ssh -o BatchMode=yes "$1" "$2"; }
snap() {
  on "$HEAD" "curl -s -m 3 $API/metrics | awk '$AWK_METRICS'
    awk '/MemAvailable/{printf \" avail=%s\", sprintf(\"%.2f\", \$2/1048576)}' /proc/meminfo
    awk '/^pswpin|^pswpout|^pgmajfault/{gsub(\"pswp\",\"swap\");printf \" %s=%s\", \$1, \$2}' /proc/vmstat
    awk '{split(\$2,a,\"=\");printf \" psi=%s\", a[2]; exit}' /proc/pressure/memory
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{printf \" gpu_util=%s\", \$1}'
    printf ' roce=%s\n' \$(cat /sys/class/infiniband/$IBDEV/ports/1/counters/port_xmit_data)"
  on "$WORKER" "awk '/MemAvailable/{printf \" w_avail=%s\", sprintf(\"%.2f\", \$2/1048576)}' /proc/meminfo
                awk '/^pgmajfault/{printf \" w_majfault=%s\", \$2}' /proc/vmstat"
}
row() { python3 -c '
import sys, time
p, c, dt = dict(l.split("=") for l in sys.argv[1].split()), dict(l.split("=") for l in sys.argv[2].split()), float(sys.argv[3])
def f(d, k):                      # counters arrive in scientific notation (4.68835e+06); parse fully
    try: return float(d.get(k) or 0)
    except ValueError: return 0.0
hit = 100 * (f(c,"ph")-f(p,"ph")) / max(1, f(c,"pq")-f(p,"pq"))
print(",".join(str(x) for x in [time.strftime("%FT%TZ", time.gmtime()), f(c,"run"), f(c,"wait"),
      f(c,"waitcap"), f(c,"waitdef"), f(c,"kv"),
      f(c,"gen")-f(p,"gen"), f(c,"comp")-f(p,"comp"), f(c,"hit")-f(p,"hit"), f(c,"preempt")-f(p,"preempt"), round(hit,1), f(c,"gpu_util"), f(c,"avail"),
      (f(c,"pswpout")-f(p,"pswpout"))*4/1024/dt, f(c,"pgmajfault")-f(p,"pgmajfault"), f(c,"psi"),
      (f(c,"roce")-f(p,"roce"))*4/1024/dt, f(c,"w_avail"), f(c,"w_majfault")-f(p,"w_majfault")]))' "$1" "$2" "$3"; }

prev=$(snap); t0=$(date +%s)
while :; do
  sleep "$INTERVAL"
  cur=$(snap); now=$(date +%s); line=$(row "$prev" "$cur" "$((now - t0))")
  echo "$line" | tee -a "$OUT"
  prev=$cur; t0=$now; [ "$INTERVAL" = 0 ] && break
done
