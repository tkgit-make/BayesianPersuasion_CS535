"""One public bit per round, starting in round 1; fair independent tie coins.
Run: python3 reproduce_cumulative_majority.py. Standard library only.
Outputs CSV and a three-panel SVG for each true state next to this script.
"""
import csv, math, random
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as E
N, T, SEED = 10000, 30, 20261117
Q = 0.7
ELL = math.log(Q / (1 - Q))
BIASES = [0, 0.25, 0.5, 0.75]
PRIORS = [(0.45, 0.55), (0.3, 0.7), (0.2, 0.8)]

def logistic(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)

def update(x, signal, bias):
    # Neutral beliefs give neither confirmation nor disconfirmation.
    weight = 1 if abs(x) < 1e-12 else (1 + bias if x * signal > 0 else 1 - bias)
    return x + weight * signal
OUT=Path(__file__).parent
def advance(weights,q,t,m):
    # Filter hidden cumulative H count by the observed public bit.
    new=[0.]*(t+1)
    for k in range(t+1):
        mass=(weights[k]*(1-q) if k<t else 0)+(weights[k-1]*q if k else 0)
        new[k]=mass*(.5 if 2*k==t else float((2*k>t)==m))
    z=sum(new)
    return z,[p/z for p in new]
def run(theta):
    rng=random.Random(SEED);coin=random.Random(SEED+1)
    paths=[Counter(),Counter()]
    for _ in range(N):
        raw=rolling=heads=0
        for t in range(1,T+1):
            bit=rng.random()<(Q if theta else 1-Q)
            heads+=bit
            message=coin.random()<.5 if 2*heads==t else 2*heads>t
            raw=2*raw+bit;rolling=2*rolling+message
        paths[0][raw]+=1;paths[1][rolling]+=1
    configs=[(p,h,b) for p,h in PRIORS for b in BIASES]
    curves={}
    for policy,counts in enumerate(paths):
        tree=[counts]
        for _ in range(T):
            parents=Counter()
            for key,v in tree[-1].items():parents[key//2]+=v
            tree.append(parents)
        tree.reverse()
        sums=[[N*(h-p)]+[0.]*T for p,h,b in configs]
        def walk(t,key,bel,w0,w1):
            if t==T:return
            for bit in [0,1]:
                child=2*key+bit;cnt=tree[t+1].get(child,0)
                if not cnt:continue
                a=c=None
                if policy:
                    p0,a=advance(w0,1-Q,t+1,bit);p1,c=advance(w1,Q,t+1,bit)
                    evidence=math.log(p1/p0)
                else:evidence=(2*bit-1)*math.log(Q/(1-Q))
                nxt=[]
                for i,((low,high),(_,__,b)) in enumerate(zip(bel,configs)):
                    l=update(low,evidence,b);h=update(high,evidence,b)
                    sums[i][t+1]+=cnt*abs(logistic(h)-logistic(l));nxt.append((l,h))
                walk(t+1,child,nxt,a,c)
        walk(0,0,[(math.log(p/(1-p)),math.log(h/(1-h))) for p,h,b in configs],[1.],[1.])
        for config,s in zip(configs,sums):curves[(policy,)+config]=[v/N for v in s]
    with (OUT/f'cumulative_theta{theta}.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['cumulative_policy','low','high','bias','round','mean_gap'])
        for key,vs in curves.items():
            for t,v in enumerate(vs):w.writerow([*key,t,v])
    svg=E.Element('svg',xmlns='http://www.w3.org/2000/svg',width='1110',height='430',viewBox='0 0 1110 430')
    def el(tag,**kw):return E.SubElement(svg,tag,{k.replace('_','-'):str(v) for k,v in kw.items()})
    el('rect',width='100%',height='100%',fill='white')
    def label(x,y,s,size=13,bold=False,**kw):
        el('text',x=x,y=y,font_family='Arial,sans-serif',font_size=size,font_weight='bold' if bold else 'normal',fill='#182438',**kw).text=s
    label(28,32,'Does publishing the cumulative majority reduce disagreement?',23,True)
    label(28,58,f'True state {theta}  ·  10,000 matched sequences  ·  70% test accuracy  ·  one public bit each round')
    colors=['#64748b','#2563eb','#c46a04','#c02635']
    for j,(p,h) in enumerate(PRIORS):
        x=65+355*j;y=105;w=300;height=220
        label(x,89,f'Priors {p:.2f} / {h:.2f}',15,True)
        for i in range(6):
            yy=y+height*(1-i/5)
            el('line',x1=x,x2=x+w,y1=yy,y2=yy,stroke='#dee4eb');label(x-8,yy+4,f'{i/5:.1f}',11,text_anchor='end')
        for t in [0,10,20,30]:label(x+t*w/T,345,str(t),11,text_anchor='middle')
        label(x+w/2,364,'Signal round',12,text_anchor='middle')
        for b,color in zip(BIASES,colors):
            for policy in [0,1]:
                points=' '.join(f'{x+w*t/T:.2f},{y+height*(1-v):.2f}' for t,v in enumerate(curves[policy,p,h,b]))
                el('polyline',points=points,fill='none',stroke=color,stroke_width=2.3,stroke_dasharray='5 4' if policy==0 else 'none')
    label(0,0,'Mean absolute belief gap',12,transform='translate(17 214) rotate(-90)',text_anchor='middle')
    for i,(b,col) in enumerate(zip(BIASES,colors)):
        x=45+120*i;el('line',x1=x,x2=x+22,y1=397,y2=397,stroke=col,stroke_width=3);label(x+28,401,f'b = {b:g}')
    for x,style,title in [(590,'5 4','Direct test report'),(835,'none','Cumulative majority')]:
        el('line',x1=x,x2=x+32,y1=397,y2=397,stroke='#172033',stroke_width=2.4,stroke_dasharray=style);label(x+40,401,title)
    E.ElementTree(svg).write(OUT/f'cumulative_theta{theta}.svg',encoding='utf-8')
    print(theta,[curves[k,.3,.7,.75][-1] for k in [0,1]],flush=True)
if __name__=='__main__':
    for theta in [0,1]:run(theta)
