from pathlib import Path
import json,html,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parents[2]
source=root/'benchmarks/kv-cache/20261118-104036/results.json'
out=root/'dev-notes/assets/2026-11-18-kv-cache';out.mkdir(parents=True,exist_ok=True)
data=json.loads(source.read_text());groups=[(m,mode,c) for m in ['Qwen3-0.6B','Qwen3-8B'] for mode in ['single','tp'] for c in ['short','long']]
colors=['#64748b','#0d9488']
for field,title,unit,better in [('tpot_ms','Time per output token','ms','Lower is better'),('output_tok_s','End-to-end output throughput','token/s','Higher is better'),('ttft_ms','Time to first token','ms','Lower is better')]:
 values=[]
 for m,mode,c in groups:
  values.append([next(x['median'][field] for x in data if (x['model'],x['mode'],x['case'],x['cache'])==(m,mode,c,flag)) for flag in [False,True]])
 maxval=max(max(v) for v in values)*1.20
 s=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="660" viewBox="0 0 1100 660" role="img">',f'<title>{title}: KV cache off versus on</title>','<rect width="1100" height="660" fill="#f8fafc"/>','<g font-family="Arial, sans-serif" fill="#0f172a">',f'<text x="32" y="40" font-size="25" font-weight="bold">KV Cache | {title}</text>',f'<text x="32" y="66" font-size="14" fill="#475569">{better} · Median of 3 measured runs after 1 warmup · FP32 · 32 output tokens</text>']
 for j,label in enumerate(['Cache OFF','Cache ON']):s += [f'<rect x="{760+j*150}" y="87" width="16" height="16" fill="{colors[j]}"/>',f'<text x="{783+j*150}" y="100" font-size="14">{label}</text>']
 left=245;width=700;top=135;row=55
 for tick in range(5):
  value=maxval*tick/4;x=left+width*tick/4
  s += [f'<line x1="{x}" y1="120" x2="{x}" y2="570" stroke="#e2e8f0"/>',f'<text x="{x}" y="592" text-anchor="middle" font-size="12">{value:.1f}</text>']
 for i,((m,mode,c),v) in enumerate(zip(groups,values)):
  y=top+i*row;label=f'{m} / {mode.upper()} / '+('20 tokens' if c=='short' else '178 tokens')
  s.append(f'<text x="230" y="{y+15}" text-anchor="end" font-size="13">{label}</text>')
  for j,value in enumerate(v):
   bar=width*value/maxval;yy=y+j*19
   s += [f'<rect x="{left}" y="{yy}" width="{bar:.2f}" height="15" rx="3" fill="{colors[j]}"/>',f'<text x="{left+bar+6:.2f}" y="{yy+12}" font-size="12">{value:.2f}</text>']
  if field=='tpot_ms':s.append(f'<text x="1020" y="{y+19}" text-anchor="middle" font-size="16" fill="#0f766e" font-weight="bold">{v[0]/v[1]:.2f}x</text>')
 s += [f'<text x="{left+width/2}" y="616" text-anchor="middle" font-size="13">{unit}</text>','<text x="32" y="645" font-size="12" fill="#64748b">max-context=512 · debug enabled in both modes · per-turn prefill · no clock locking / GPU isolation</text>','</g></svg>']
 text='\n'.join(s);ET.fromstring(text);(out/(field+'.svg')).write_text(text)
p=root/'dev-notes/2026-11-18-kv-cache.md';s=p.read_text().split('\n## 性能提升可视化')[0].rstrip()+'\n';s+='''
## 性能提升可视化

数据来自同一二进制的缓存开关对照：FP32、max-context=512、输出32 token，每组预热1次后测3次取中位数；短/长输入分别为20/178 token。各配置生成文本一致。横条旁为实测数值，TPOT 图右侧为无缓存/有缓存的加速比。

### 生成间隔 TPOT（越低越好）

![KV Cache TPOT 对比](assets/2026-11-18-kv-cache/tpot_ms.svg)

### 端到端输出吞吐（越高越好）

![KV Cache 输出吞吐对比](assets/2026-11-18-kv-cache/output_tok_s.svg)

吞吐包含首次 prefill，不能直接当作纯 decode 吞吐。

### 首 token 延迟 TTFT（越低越好）

![KV Cache 首 token 延迟对比](assets/2026-11-18-kv-cache/ttft_ms.svg)

缓存主要降低后续生成的重复计算；首 token 仍需 prefill，热态 TTFT 大体接近。未锁频或隔离其他负载，微小差异不作收益结论。原始数据和完整运行条件见前文性能归档。

复现图表：`python3 dev-notes/scripts/plot_kv_cache.py`。图表为无外部依赖的 SVG，可直接在 GitHub 开发记录中显示。
''';p.write_text(s)
print('Generated 3 SVG charts and updated development record')
