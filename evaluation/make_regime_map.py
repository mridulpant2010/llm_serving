import sys
# usage: python evaluation/make_regime_map.py [results_dir] [output_dir]
U = (sys.argv[1] if len(sys.argv) > 1 else "results_budget").rstrip("/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else "evaluation/output").rstrip("/") + "/"
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle, Patch

FP16,FP8="#2a78d6","#eb6834"; DRAFT,EAGLE,NGRAM="#6250d6","#1baf7a","#c98500"; INK,INK2="#0f172a","#52514e"
plt.rcParams.update({"font.family":"Inter","font.size":15,"figure.facecolor":"white","savefig.facecolor":"white","text.color":INK})
rows=[json.loads(l) for l in open(U+"results.jsonl") if l.strip()]
df=pd.DataFrame([r for r in rows if r["status"]=="ok"]); corp=df[df.workload=="corpus"]
Cs=[1,4,16,64,256]; meths=[("draft_model","Draft model (1B)"),("eagle","EAGLE"),("ngram","N-gram")]
S={}; LO={}; HI={}
for kv in ["auto","fp8"]:
    base=corp[(corp.kv_dtype==kv)&(corp.spec=="none")]
    for sp,_ in meths:
        d=corp[(corp.kv_dtype==kv)&(corp.spec==sp)]
        for c in Cs:
            v=[r.throughput_tps/base[(base.concurrency==c)&(base.rep==r.rep)].throughput_tps.iloc[0] for _,r in d[d.concurrency==c].iterrows()]
            S[(kv,sp,c)]=np.mean(v); LO[(kv,sp,c)]=min(v); HI[(kv,sp,c)]=max(v)
cmap=LinearSegmentedColormap.from_list("div",[FP8,"#f3f4f6",FP16]); norm=TwoSlopeNorm(vmin=.7,vcenter=1.0,vmax=1.45)
fig=plt.figure(figsize=(13.33,7.5))
fig.suptitle("Regime map: speculation pays at low concurrency and loses from C=64",x=.045,y=.985,ha="left",fontsize=22,fontweight="bold")
fig.text(.045,.915,"Throughput speedup vs. no speculation (same KV dtype), ctx 2,048 · dotted = within repeat noise of parity",fontsize=14.5,color=INK2)
for k,(kv,title,pos) in enumerate([("auto","FP16 KV cache",[.13,.44,.38,.40]),("fp8","FP8 KV cache",[.57,.44,.38,.40])]):
    ax=fig.add_axes(pos); ax.set_xlim(0,5); ax.set_ylim(0,3); ax.invert_yaxis()
    for i,(sp,nm) in enumerate(meths):
        for j,c in enumerate(Cs):
            v=S[(kv,sp,c)]; col=cmap(norm(v)); ax.add_patch(Rectangle((j,i),1,1,facecolor=col,edgecolor="white",lw=3))
            par=LO[(kv,sp,c)]<=1.0<=HI[(kv,sp,c)]
            if par: ax.add_patch(Rectangle((j+.06,i+.06),.88,.88,fill=False,edgecolor=INK,lw=1.6,ls=(0,(2,2))))
            lum=.299*col[0]+.587*col[1]+.114*col[2]
            ax.text(j+.5,i+.5,f"{'≈' if par else ''}{v:.2f}×",ha="center",va="center",fontsize=16,fontweight="bold",color="white" if lum<.5 else INK)
    ax.set_xticks([j+.5 for j in range(5)]); ax.set_xticklabels(Cs,fontsize=15); ax.set_xlabel("")
    fig.text(pos[0]+pos[2]/2,.375,"Concurrency (C)",ha="center",fontsize=14,color=INK2)
    ax.set_yticks([i+.5 for i in range(3)]); ax.set_yticklabels([n for _,n in meths] if k==0 else ["","",""],fontsize=15)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(title,loc="left",fontsize=18,fontweight="bold",color=FP16 if kv=="auto" else FP8,pad=10)
# colourbar legend
cax=fig.add_axes([.13,.31,.38,.02]); grad=np.linspace(.7,1.45,256)[None,:]
cax.imshow(grad,aspect="auto",cmap=cmap,norm=norm,extent=[.7,1.45,0,1]); cax.set_yticks([]); cax.set_xticks([.7,.85,1.0,1.2,1.45]); cax.set_xticklabels(["0.70× loses","0.85×","1.0× parity","1.2×","1.45× wins"],fontsize=12.5,color=INK2)
cax.tick_params(length=3); [s.set_visible(False) for s in cax.spines.values()]
# winners strip
cfg={}
sp_ab=df[df.stage.isin(["stage3_spec_fp16","stage4_spec_fp8"])]
ag=sp_ab.groupby(["kv_dtype","spec","concurrency"]).throughput_tps.mean()
fig.text(.57,.30,"Best configuration at each C (highest tok/s)",fontsize=13.5,fontweight="bold",color=INK2)
ax2=fig.add_axes([.57,.10,.38,.17]); ax2.set_xlim(0,5); ax2.set_ylim(0,1); ax2.axis("off")
NM={"none":"no spec","draft_model":"+ draft","eagle":"+ EAGLE","ngram":"+ N-gram"}; CC={"draft_model":DRAFT,"eagle":EAGLE,"ngram":NGRAM}
for j,c in enumerate(Cs):
    kv,sp=max(ag.index.droplevel(2).unique(),key=lambda t:ag[(t[0],t[1],c)])
    col=CC.get(sp,FP8 if kv=="fp8" else FP16)
    ax2.text(j+.5,.5,f"{'FP8' if kv=='fp8' else 'FP16'}\n{NM[sp]}",ha="center",va="center",fontsize=13,fontweight="bold",color="white",bbox=dict(boxstyle="round,pad=.4",fc=col,ec="none"))
fig.text(.13,.235,"Reading the map",fontsize=14,fontweight="bold",color=INK)
fig.text(.13,.045,"• Speculation pays only at C ≤ 4 (≤ 16 for the draft model).\n• FP8 shifts N-gram and EAGLE from losing to parity at C=16.\n• From C=64 up, no speculative method beats no speculation.",fontsize=13,color=INK2,va="bottom",linespacing=1.5)
fig.savefig(OUT+"11_regime_map.png",dpi=200); print({k:round(v,2) for k,v in S.items() if k[2] in (16,)})
