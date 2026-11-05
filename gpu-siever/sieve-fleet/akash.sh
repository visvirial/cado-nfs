#!/bin/bash
# Run sieving clients on Akash through the Console API (managed wallet, paid from the account's
# USD credits), next to the Vast fleet.
#
#   akash.sh prices                      GPU models: available/total, providers, $/h min/med/max
#   akash.sh create MODEL [IMAGE]        a deployment of one GPU of MODEL (rtx5090, a100, h100,
#                                        v100, pro6000se, ...) running host/container-boot.sh; prints
#                                        the dseq
#   akash.sh bids DSEQ                   its bids, cheapest first: provider, $/h, GPU
#   akash.sh lease DSEQ [PROVIDER]       accept the bid of PROVIDER (default: the cheapest)
#   akash.sh run MODEL [IMAGE]           create, wait for bids, lease the cheapest; prints dseq
#   akash.sh list                        the deployments, with their leases and $/h
#   akash.sh close DSEQ...               close deployments (the unspent deposit comes back)
#
# The API key is read from ~/.config/akash/api_key; SERVER, CERTSHA1, MAKEFB and MAKEFB_OUT
# from $FLEET_DIR/fleet.conf. The container fetches commit SRC_COMMIT (default:
# origin/rsa-270, so it must be pushed) from GitHub, as on SaladCloud; its client id is
# vast-svsalad-akash-<model>-<dseq>. Prices: a bid is in uact per block, 1 ACT = 1 USD, and a
# block takes about 5.88 s (AKASH_BLOCK_S, measured on 2026-11-06), so $/h = amount * 3600 / AKASH_BLOCK_S / 1e6. The
# sm_70 to sm_90 GPUs take the CUDA 12.4.1 image (glas built with 12.8 took 20% longer on an
# A100), the others 12.8.1. A deployment asks for AKASH_CPU (16) CPUs and AKASH_MEMORY (32Gi).
set -eu
HERE=$(dirname "$(readlink -f "$0")"); . "$HERE/env.sh"; set -a; . "$FLEET_DIR/fleet.conf"; set +a
K=$(cat ~/.config/akash/api_key); A=https://console-api.akash.network
BLOCK_S=${AKASH_BLOCK_S:-5.88}
api() { m=$1; u=$2; shift 2; curl -sS -m 60 -X $m -H "x-api-key: $K" -H "Content-Type: application/json" "$A$u" "$@"; }
image() { case $1 in v100|a100|a10g|h100|h200|t4) echo nvidia/cuda:12.4.1-devel-ubuntu22.04;; *) echo nvidia/cuda:12.8.1-devel-ubuntu22.04;; esac; }
create() { model=$1; img=${2:-$(image $1)}; c=${SRC_COMMIT:-$(git -C "$HERE" rev-parse origin/rsa-270)}
  body=$(python3 - "$model" "$img" "$c" "$HERE/host/container-boot.sh" <<'PY'
import json,os,sys
model,img,c,boot=sys.argv[1:]
env={"SRC_COMMIT":c,"SERVER":os.environ["SERVER"],"CERTSHA1":os.environ["CERTSHA1"],
     "MAKEFB":os.environ.get("MAKEFB",""),"MAKEFB_OUT":os.environ.get("MAKEFB_OUT",""),
     "SALAD_MACHINE_ID":"akash-"+model}
# JSON is YAML, so the SDL is written as JSON
sdl={"version":"2.0",
  # providers refuse a deployment that exposes no global port; nothing listens on this one
  "services":{"sieve":{"image":img,"command":["bash","-c"],"args":[open(boot).read()],
    "expose":[{"port":8080,"as":80,"to":[{"global":True}]}],
    "env":["%s=%s"%kv for kv in env.items()]}},
  "profiles":{"compute":{"sieve":{"resources":{"cpu":{"units":int(os.environ.get("AKASH_CPU","16"))},"memory":{"size":os.environ.get("AKASH_MEMORY","32Gi")},
      "storage":[{"size":"60Gi"}],"gpu":{"units":1,"attributes":{"vendor":{"nvidia":[{"model":model}]}}}}}},
    "placement":{"dcloud":{"pricing":{"sieve":{"denom":"uact","amount":int(os.environ.get("AKASH_MAXPRICE","10000"))}}}}},
  "deployment":{"sieve":{"dcloud":{"profile":"sieve","count":1}}}}
print(json.dumps({"data":{"sdl":json.dumps(sdl)}}))
PY
)
  api POST /v1/deployments --data "$body" | python3 -c 'import json,sys
d=json.load(sys.stdin); x=d.get("data") or {}
if not x.get("dseq"): sys.exit("create failed: %s" % json.dumps(d)[:400])
print(x["dseq"])'; }
bids() { api GET "/v1/bids?dseq=$1" | python3 -c '
import json,sys
b=sys.argv[1]; v=[]
for x in json.load(sys.stdin).get("data") or []:
    bid=x["bid"]
    if bid.get("state")!="open": continue
    p=float(bid["price"]["amount"])*3600/float(b)/1e6
    g=[r["resources"].get("gpu",{}) for r in bid.get("resources_offer",[])]
    attrs=[a.get("key","")+"="+a.get("value","") for gg in g for a in (gg.get("attributes") or [])]
    v.append((p,bid["id"]["provider"],bid["id"]["gseq"],bid["id"]["oseq"],bid["price"]["denom"]," ".join(attrs)))
for r in sorted(v): print("%s %.3f %s %s %s %s"%(r[1],r[0],r[2],r[3],r[4],r[5]))' $BLOCK_S; }
lease() { dseq=$1; prov=${2:-}
  line=$(bids $dseq | { [ -n "$prov" ] && grep "^$prov " || cat; } | head -1)
  [ -n "$line" ] || { echo "no open bid for $dseq" >&2; return 1; }
  set -- $line
  api POST /v1/leases --data "{\"leases\":[{\"dseq\":\"$dseq\",\"gseq\":$3,\"oseq\":$4,\"provider\":\"$1\"}]}" | python3 -c '
import json,sys
d=json.load(sys.stdin); x=d.get("data") or {}
l=x.get("leases") or []
print("lease", (l[0].get("state") if l else json.dumps(d)[:300]))'
  echo "$dseq $1 \$$2/h" ; }
case ${1:-} in
prices) api GET /v1/gpu-prices | python3 -c '
import json,sys
for m in sorted(json.load(sys.stdin)["models"], key=lambda m:(m.get("price") or {}).get("min",99)):
    p=m.get("price") or {}
    print("%-10s %-6s %-5s avail %3d/%3d providers %d/%d  $/h %.2f %.2f %.2f"%(m["model"],m["ram"],m["interface"],
      m["availability"]["available"],m["availability"]["total"],m["providerAvailability"]["available"],
      m["providerAvailability"]["total"],p.get("min",0),p.get("med",0),p.get("max",0)))';;
create) create "$2" "${3:-}";;
bids) bids "$2";;
lease) lease "$2" "${3:-}";;
run) dseq=$(create "$2" "${3:-}"); echo "dseq $dseq" >&2
  for i in $(seq 40); do sleep 3; [ -n "$(bids $dseq)" ] && break; done; sleep 15
  bids $dseq >&2; lease $dseq || { echo "closing $dseq" >&2; api DELETE /v1/deployments/$dseq > /dev/null; exit 1; };;
list) api GET "/v1/deployments?limit=100" | python3 -c '
import json,sys
b=float(sys.argv[1])
for d in json.load(sys.stdin)["data"]["deployments"]:
    dep=d.get("deployment",{}); ls=d.get("leases") or []
    for l in ls or [{}]:
        pr=l.get("price") or {}
        print(dep.get("id",{}).get("dseq"), dep.get("state"), l.get("id",{}).get("provider","-"), l.get("state","-"),
              "$%.3f/h"%(float(pr.get("amount",0))*3600/b/1e6))' $BLOCK_S;;
close) shift; for d in "$@"; do echo "$d $(api DELETE /v1/deployments/$d)"; done;;
*) sed -n '2,24p' "$0"; exit 1;;
esac
