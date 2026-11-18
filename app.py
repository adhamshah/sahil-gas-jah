import json
import os
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Your Hugging Face Space (set HF_SPACE_URL / HF_TOKEN as Vercel env vars if you want to override)
SPACE_URL = os.getenv("HF_SPACE_URL", "https://adham-shah-sahil-gas-jah.hf.space").rstrip("/")
HF_TOKEN = os.getenv("HF_TOKEN")  # only needed if the Space is private

app = FastAPI(title="Sahil Gas Jah AI")


class PredictRequest(BaseModel):
    model: Literal["RNN", "GRU", "LSTM"] = "LSTM"
    history: list[float] = Field(min_length=7, max_length=7)  # last 7 days of sales, oldest first
    last_date: str                                             # date of the 7th (latest) day, YYYY-MM-DD
    shortage: bool = False
    ramadan: bool = False
    eid: bool = False


@app.post("/api/predict")
async def predict(r: PredictRequest):
    if any(v < 0 or v > 500 for v in r.history):
        raise HTTPException(400, "Sales values must be between 0 and 500.")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    payload = {"data": [r.model, ",".join(str(v) for v in r.history), r.last_date,
                        r.shortage, r.ramadan, r.eid]}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            s = await client.post(f"{SPACE_URL}/gradio_api/call/predict", json=payload, headers=headers)
            s.raise_for_status()
            event_id = s.json()["event_id"]
            res = await client.get(f"{SPACE_URL}/gradio_api/call/predict/{event_id}", headers=headers)
            res.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Model service error: {e}")
    event, data = None, None
    for line in res.text.splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:") and event == "complete":
            data = json.loads(line[5:].strip())
    if not data:
        raise HTTPException(502, "Model service returned an error.")
    return {"model": r.model, "date": data[0]["date"], "cylinders": data[0]["cylinders"]}


def cyl(w, h, body, cap):
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 60 120" aria-hidden="true">'
            f'<rect x="22" y="2" width="16" height="12" rx="3" fill="{cap}"/>'
            f'<rect x="26" y="10" width="8" height="10" fill="{cap}"/>'
            f'<rect x="6" y="18" width="48" height="98" rx="22" fill="{body}"/>'
            f'<rect x="6" y="58" width="48" height="9" fill="#fff" fill-opacity=".35"/></svg>')


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Sahil Gas Jah AI</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box}
body{margin:0;background:#E8ECEE;color:#17262E;font-family:Barlow,system-ui,sans-serif;font-size:16px}
h1,h2{font-family:"Barlow Condensed",sans-serif;margin:0}
.wrap{max-width:1200px;margin:0 auto;padding:max(20px,env(safe-area-inset-top)) clamp(14px,4vw,40px) 40px}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:24px}
.brand{display:flex;align-items:center;gap:14px}
h1{font-size:clamp(34px,8vw,48px);line-height:1;font-weight:700}
.sub{margin:6px 0 0;font-size:clamp(15px,3.8vw,17px)}
.deco{display:flex;gap:10px;align-items:flex-end}
.layout{display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start}
.main{flex:999 1 520px;min-width:0}.side{flex:1 1 300px;min-width:0;display:flex;flex-direction:column;gap:20px}
.card{background:#fff;border-radius:10px;padding:clamp(16px,4vw,28px);box-shadow:0 0 0 2px #17262E}
h2{font-size:26px;font-weight:700;margin:0 0 12px}.card h2:not(:first-child){margin-top:24px}
.days{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:10px}
.day label{display:block;font-weight:600;font-size:14px;margin-bottom:4px;text-align:center}
.bar{height:64px;display:flex;align-items:flex-end;margin-bottom:6px}.bar div{width:100%;background:#9DB4C0;border-radius:4px 4px 0 0;height:0;transition:height .2s}
input{font:inherit;color:#17262E}
input.n,input.d{width:100%;text-align:center;font-size:22px;font-weight:600;padding:10px 0;border:2px solid #17262E;border-radius:6px;background:#fff;min-height:48px}
input.d{font-size:18px;padding:10px;max-width:240px;text-align:left}
.row{display:flex;gap:10px;flex-wrap:wrap}
.chip{font:inherit;border:2px solid #17262E;background:#fff;color:#17262E;padding:10px 18px;border-radius:6px;cursor:pointer;min-height:48px}
.chip.on{background:#C8322B;border-color:#C8322B;color:#fff}
.go{width:100%;margin-top:26px;font:700 24px "Barlow Condensed",sans-serif;background:#17262E;color:#fff;border:0;border-radius:8px;min-height:56px;cursor:pointer}
.go:disabled{opacity:.6}
.chip:focus-visible,.go:focus-visible,input:focus-visible{outline:3px solid #E0A526;outline-offset:2px}
.res{background:#C8322B;color:#fff}.res p{margin:0}
.big{font:700 clamp(88px,26vw,120px)/1 "Barlow Condensed",sans-serif;margin:4px 0!important}
.err{background:#fff;color:#8c1d18;border-radius:8px;padding:10px 12px;margin-top:12px;font-weight:600}
.facts p{margin:0 0 8px}.facts p:last-child{margin:0}
@media (max-width:640px){
 .days{grid-template-columns:repeat(4,minmax(0,1fr))}
 .deco{display:none}
 .card h2{font-size:24px}
}
@media (prefers-reduced-motion:reduce){.bar div{transition:none}}
</style></head><body><div class="wrap">
<header><div class="brand">__LOGO__<div><h1>Sahil Gas Jah AI</h1>
<p class="sub">Tell us the last 7 days. We tell you how many cylinders to stock tomorrow.</p></div></div>
<div class="deco">__D1____D2____D3__</div></header>
<div class="layout"><div class="main card">
<h2>Cylinders sold, last 7 days</h2>
<div class="row" style="margin-bottom:14px;align-items:center"><label for="last" style="font-weight:600">Last day entered</label><input class="d" id="last" type="date"></div>
<div class="days" id="days"></div>
<h2>Model</h2><div class="row" id="models"></div>
<h2>Tomorrow is special?</h2>
<div class="row"><button class="chip" id="sh" aria-pressed="false">Supply shortage</button><button class="chip" id="ra" aria-pressed="false">Ramadan</button><button class="chip" id="ed" aria-pressed="false">Eid week</button></div>
<button class="go" id="go">Predict tomorrow</button><div class="err" id="err" hidden role="alert"></div></div>
<div class="side"><div class="card res" aria-live="polite"><p id="when" style="font-size:18px">Enter your sales and press Predict</p>
<p class="big" id="pred">–</p><p id="unit" style="font-size:20px;font-weight:600">cylinders</p><p id="buf" style="margin-top:14px;font-size:14px"></p></div>
<div class="card facts"><h2>What 3 years of data show</h2>
<p>Fridays are slowest: 32 sold on average. Saturdays are busiest: 45.</p>
<p>Shortage days drop sales to about 20, from 41.</p>
<p>Ramadan lifts sales to 46.5; the days around Eid dip to 36.</p></div></div></div></div>
<script>
const $=id=>document.getElementById(id),WD=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
let model='LSTM',flags={sh:false,ra:false,ed:false};
const iso=d=>d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');
const parse=s=>{const [y,m,d]=s.split('-').map(Number);return new Date(y,m-1,d)};
const add=(d,n)=>{const x=new Date(d);x.setDate(x.getDate()+n);return x};
$('last').value=iso(new Date());
const box=$('days');
for(let i=0;i<7;i++){box.insertAdjacentHTML('beforeend',`<div class="day"><div class="bar"><div></div></div><label for="v${i}" id="l${i}"></label><input class="n" id="v${i}" type="number" inputmode="numeric" min="0" max="71" placeholder="0"></div>`);
 $('v'+i).addEventListener('input',bars)}
function bars(){for(let i=0;i<7;i++){const v=Math.min(71,Math.max(0,+$('v'+i).value||0));box.children[i].querySelector('.bar div').style.height=(v/71*100)+'%'}}
function labels(){const end=parse($('last').value||iso(new Date()));for(let i=0;i<7;i++)$('l'+i).textContent=WD[add(end,i-6).getDay()]}
$('last').addEventListener('change',labels);labels();
['RNN','GRU','LSTM'].forEach(k=>{const b=document.createElement('button');b.className='chip'+(k===model?' on':'');b.textContent=k;b.setAttribute('aria-pressed',k===model);
 b.onclick=()=>{model=k;[...$('models').children].forEach(c=>{const on=c.textContent===k;c.classList.toggle('on',on);c.setAttribute('aria-pressed',on)})};$('models').appendChild(b)});
for(const k of ['sh','ra','ed'])$(k).onclick=()=>{flags[k]=!flags[k];$(k).classList.toggle('on',flags[k]);$(k).setAttribute('aria-pressed',flags[k])};
$('go').onclick=async()=>{
 const err=$('err');err.hidden=true;
 const hist=[];for(let i=0;i<7;i++){const t=$('v'+i).value;if(t===''||isNaN(+t)||+t<0||+t>71){err.textContent='Please enter a sales number (0 to 71) for every day.';err.hidden=false;return}hist.push(+t)}
 if(!$('last').value){err.textContent='Pick the last day you entered.';err.hidden=false;return}
 const b=$('go');b.disabled=true;b.textContent='Asking the model…';
 try{
  const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model,history:hist,last_date:$('last').value,shortage:flags.sh,ramadan:flags.ra,eid:flags.ed})});
  const j=await r.json();if(!r.ok)throw new Error(j.detail||'Something went wrong');
  const d=parse(j.date);
  $('when').textContent='Expected on '+WD[d.getDay()]+' '+d.getDate()+' '+d.toLocaleString('en',{month:'short'})+' '+d.getFullYear();
  $('pred').textContent=j.cylinders;$('unit').textContent='cylinders · '+j.model+' model';
  $('buf').textContent='With a 10% safety buffer: '+Math.ceil(j.cylinders*1.1)+' cylinders';
 }catch(e){err.textContent=String(e.message||e);err.hidden=false}
 b.disabled=false;b.textContent='Predict tomorrow';
};
bars();
</script></body></html>"""

PAGE = (PAGE.replace("__LOGO__", cyl(44, 88, "#C8322B", "#17262E"))
            .replace("__D1__", cyl(30, 60, "#C8322B", "#17262E"))
            .replace("__D2__", cyl(36, 72, "#2F6F8F", "#17262E"))
            .replace("__D3__", cyl(42, 84, "#E0A526", "#17262E")))


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
