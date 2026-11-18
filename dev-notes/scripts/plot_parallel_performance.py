from pathlib import Path
import json,xml.etree.ElementTree as ET
r=Path(__file__).resolve().parents[2]
a=json.loads((r/'benchmarks/baselines/2026-11-17-no-kv/results.json').read_text())
b=json.loads((r/'benchmarks/kv-cache/20261118-104036/results.json').read_text())
assets=r/'dev-notes/assets/2026-11-18-parallel';assets.mkdir(parents=True,exist_ok=True)
def chart(rows,name,title):
 s=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="640" viewBox="0 0 1000 640" role="img">',f'<title>{title}</title>','<rect width="1000" height="640" fill="#f8fafc"/>','<g font-family="Arial,sans-serif" fill="#0f172a">',f'<text x="28" y="38" font-size="24" font-weight="bold">{title}</text>','<text x="28" y="65" font-size="14">TPOT (ms) - lower is better; ratio = single-GPU TPOT / mode TPOT</text>']
 maximum=max(x[1] for x in rows)*1.16
 for i,(label,value,ratio,mode) in enumerate(rows):
  y=100+i*60;w=540*value/maximum;color={'single':'#64748b','pp':'#d97706','tp':'#0d9488'}[mode]
  s += [f'<text x="290" y="{y+19}" text-anchor="end" font-size="14">{label}</text>',f'<rect x="305" y="{y}" width="{w:.2f}" height="28" rx="4" fill="{color}"/>',f'<text x="{312+w:.2f}" y="{y+20}" font-size="13">{value:.3f}</text>',f'<text x="925" y="{y+20}" text-anchor="middle" font-size="16">{ratio:.2f}x</text>']
 s += ['<text x="28" y="610" font-size="13">Ratios below 1 mean regression. No clock locking / GPU isolation.</text>','</g></svg>'];text='\n'.join(s);ET.fromstring(text);(assets/name).write_text(text)
rows=[]
for case in ['short','long']:
 ref=next(x['median']['tpot_ms'] for x in a if x['case']==case and x['mode']=='single')
 for mode in ['single','pp','tp']:
  v=next(x['median']['tpot_ms'] for x in a if x['case']==case and x['mode']==mode);rows.append((f'{case} / {mode.upper()}',v,ref/v,mode))
chart(rows,'06b-pp-tp.svg','Qwen3-0.6B | Single vs PP vs TP | Cache OFF')
rows=[]
for cache in [False,True]:
 for case in ['short','long']:
  ref=next(x['median']['tpot_ms'] for x in b if x['model']=='Qwen3-8B' and x['cache']==cache and x['case']==case and x['mode']=='single')
  for mode in ['single','tp']:
   v=next(x['median']['tpot_ms'] for x in b if x['model']=='Qwen3-8B' and x['cache']==cache and x['case']==case and x['mode']==mode);rows.append((f'Cache {"ON" if cache else "OFF"} / {case} / {mode.upper()}',v,ref/v,mode))
chart(rows,'8b-tp.svg','Qwen3-8B | Single vs TP | Compare within cache mode')
lines=['# 2026-11-18：补充 PP / TP 性能对比','','本记录整理已归档数据，不新增性能测试。分别在同组内比较，不能跨日期/输入/缓存状态混算并行收益。速度比定义为单卡 TPOT / 当前模式 TPOT，大于1为加速，小于1为退化。','','## 0.6B：单卡、PP、TP（无 KV Cache）','','来源：[2026-11-17 原始基线](../benchmarks/baselines/2026-11-17-no-kv/README.md)。FP32，上下文256，每种配置独立加载；每个输入预热1次后测5次，取中位数，轮间清空历史。短输入23 token、输出32；长输入179 token、输出19。长短输入之间不直接比较吞吐，模式间仅比较相同输入/输出组。','','|输入|模式|TTFT ms|TPOT ms|输出 token/s|相对单卡 TPOT 速度比|','|---|---|---:|---:|---:|---:|']
for x in a:
 m=x['median'];ref=next(y['median']['tpot_ms'] for y in a if y['case']==x['case'] and y['mode']=='single');lines.append(f"|{x['case']}|{x['mode']}|{m['ttft_ms']:.3f}|{m['tpot_ms']:.3f}|{m['output_tok_s']:.3f}|{ref/m['tpot_ms']:.2f}x|")
lines+=['','![0.6B PP TP 对比](assets/2026-11-18-parallel/06b-pp-tp.svg)','','结论：该单请求测试中PP没有加速；TP短输入退化，长输入TPOT仅约1.03x。长输入的小幅差异不足以证明稳定收益，需要更多重复和负载控制。当前PP为顺序分层执行，不是多请求微批流水线，因此不能据此判断生产并发PP的收益。','','## 8B：同缓存状态下单卡与 TP','','来源：[KV Cache 开关实验](../benchmarks/kv-cache/20261118-104036/README.md)。同一二进制、FP32、max-context=512，短/长输入20/178 token，均实际输出32 token；每组预热1次后测3次取中位数，debug开启。只在缓存状态相同的两行之间计算TP收益。','','|缓存|输入|单卡 TPOT ms|TP TPOT ms|TP 加速比|','|---|---|---:|---:|---:|']
for cache in [False,True]:
 for case in ['short','long']:
  v=[next(x['median']['tpot_ms'] for x in b if (x['model'],x['mode'],x['case'],x['cache'])==('Qwen3-8B',mode,case,cache)) for mode in ['single','tp']];lines.append(f"|{'开启' if cache else '关闭'}|{case}|{v[0]:.3f}|{v[1]:.3f}|{v[0]/v[1]:.2f}x|")
lines+=['','![8B TP 对比](assets/2026-11-18-parallel/8b-tp.svg)','','结论：8B无缓存长输入TP约1.63x，开启缓存后长输入TP约1.18x。缓存降低重复计算后，两卡相对收益变小；具体瓶颈比例尚未profiling，不将这一现象当作纯通信耗时测量。','','## 缺口与解释边界','','- 8B PP 尚未收集同口径正式性能数据，不能补写为已验证收益。','- 图表不是新的跑分；历史数据未锁频或隔离其他GPU进程，样本量较小。','- TTFT、TPOT、整轮吞吐分别解释；当前数据不含高并发服务吞吐、同时双向链路或多请求PP调度。','- KV Cache开关收益见对应记录；本页TP收益固定缓存开关，二者不混用。','','复现本页和图表：`python3 dev-notes/scripts/plot_parallel_performance.py`。']
(r/'dev-notes/2026-11-18-parallel-performance.md').write_text('\n'.join(lines)+'\n')
print('parallel performance notes and charts generated')
