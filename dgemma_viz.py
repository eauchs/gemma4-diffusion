#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dgemma_viz.py — oscilloscope de débruitage pour DiffusionGemma 26B-A4B (MLX).

But : voir REELLEMENT le sampler tourner. A chaque step de denoising on stream
l'état complet du canvas (256 positions) vers une UI : token courant, état
(masque / résolu / re-bruité) et confiance par position. Tu lis d'un coup d'oeil
si ça converge proprement, combien de steps, et si ça thrash aux frontières de blocs.

Lancer :
    python dgemma_viz.py --mock                                          # UI seule, dynamique simulée
    python dgemma_viz.py --model mlx-community/diffusiongemma-26B-A4B-it-4bit   # vrai modèle

Puis ouvre http://127.0.0.1:8765

Zéro dépendance hors stdlib en mode --mock. En mode réel : mlx_vlm (que t'as déjà
dans .venv ; c'est lui qui porte le sampler diffusion, PAS mlx_lm).

Le backend réel (mlx_diffusion_steps) pilote une copie INSTRUMENTÉE de la boucle
entropy-bound de mlx_vlm/generate/diffusion.py : on réutilise les helpers internes
du module et on yield l'état du canvas à chaque step. C'est le vrai sampler, pas
une approximation — argmax, entropie, masque d'acceptation et self-conditioning
sont ceux de la lib.
"""

import argparse
import json
import math
import random
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

CANVAS = 256          # taille d'un canvas DiffusionGemma
MASK = "·"            # glyphe affiché pour une position masquée

# ----------------------------------------------------------------------------
# Contrat envoyé à l'UI, un dict sérialisable par step :
#   step       : index du step de denoising (global, tous blocs confondus)
#   max_steps  : budget de steps par bloc
#   block      : index du bloc courant (0..)
#   cells      : liste de 256 {t:str, s:int, c:float}
#                s = 0 masque, 2 résolu/stable
#                c = confiance [0..1] (proba top-1 du modèle à cette position)
#   newly      : nb tokens nouvellement résolus à ce step
#   renoised   : nb tokens re-bruités (résolus -> remasqués) à ce step
#   committed  : nb total de tokens committés (blocs précédents)
#   mean_entropy : entropie moyenne sur les positions encore masquées (nats)
#   text       : texte décodé courant (committé + résolus du canvas en cours)
# ----------------------------------------------------------------------------

S_MASK, S_NEW, S_STABLE, S_RENOISE, S_COMMIT = 0, 1, 2, 3, 4


# ============================================================================
# BACKEND MOCK — reproduit la dynamique (masque -> résolu, re-bruitage,
# early-stop par entropie). Valide l'UI sans charger 26B de poids.
# ============================================================================
def mock_diffusion_steps(prompt, max_steps=48, n_blocks=2, seed=None):
    rng = random.Random(seed if seed is not None else time.time())

    pool = ("le sampler résout le canvas en parallèle puis recommence sur le bloc "
            "suivant chaque step fige les tokens les plus confiants et re bruite "
            "ceux dont l entropie reste haute jusqu à convergence ou early stop "
            "voilà pourquoi tu vois certaines cases clignoter en rouge avant de "
            "se stabiliser en vert phosphore comme un signal qui se débruite").split()

    committed_text = []
    for block in range(n_blocks):
        target_n = rng.randint(70, 130)
        truth = [pool[(block * 137 + i) % len(pool)] for i in range(target_n)]
        order = list(range(target_n))
        rng.shuffle(order)

        resolved = {}
        cursor = 0
        for step in range(max_steps):
            newly = renoised = 0
            for pos in list(resolved.keys()):
                tok, conf = resolved[pos]
                if conf < 0.55 and rng.random() < 0.30:
                    del resolved[pos]
                    renoised += 1
            budget = max(1, int((18 - step * 0.25) * (1 - step / (max_steps * 1.4))))
            picked = 0
            while picked < budget and cursor < target_n:
                pos = order[cursor]
                cursor += 1
                if pos in resolved:
                    continue
                conf = min(0.99, max(0.05, rng.gauss(0.55 + step * 0.012, 0.18)))
                resolved[pos] = (truth[pos], conf)
                newly += 1
                picked += 1

            n_masked = target_n - len(resolved)
            mean_h = (math.log(50) * n_masked / max(1, target_n)) * (1 - step / max_steps)
            cells = _mock_cells(target_n, resolved)
            cur_text = " ".join(committed_text + [resolved[i][0]
                                                  for i in range(target_n)
                                                  if i in resolved])
            yield dict(type="step", step=step, max_steps=max_steps, block=block,
                       cells=cells, newly=newly, renoised=renoised,
                       committed=len(committed_text), mean_entropy=round(mean_h, 3),
                       text=cur_text)
            time.sleep(0.06)
            if n_masked == 0 and renoised == 0:
                break
        committed_text.extend(truth)

    yield dict(type="done", text=" ".join(committed_text))


def _mock_cells(target_n, resolved):
    cells = []
    for i in range(CANVAS):
        if i < target_n and i in resolved:
            tok, conf = resolved[i]
            cells.append(dict(t=tok, s=S_STABLE, c=round(conf, 3)))
        elif i < target_n:
            cells.append(dict(t=MASK, s=S_MASK, c=0.0))
        else:
            cells.append(dict(t="", s=S_MASK, c=0.0))
    return cells


# ============================================================================
# BACKEND MLX RÉEL — boucle de denoising entropy-bound instrumentée.
# ----------------------------------------------------------------------------
# DiffusionGemma : encoder causal qui prefill le prompt dans le KV-cache, puis
# denoising bidirectionnel sur un canvas de 256 tokens. Par step le sampler garde
# les tokens dont l'entropie tient sous entropy_bound et re-bruite le reste.
# Défauts checkpoint : max_denoising_steps=48, temp 0.4->0.8 linéaire (schedule),
# entropy_bound=0.1, early-stop stable+confiant.
#
# On reproduit fidèlement le coeur de stream_diffusion_generate() en réutilisant
# les helpers internes du module, et on yield l'état du canvas à chaque step.
# ============================================================================
_MODEL = None
_PROC = None
_TOK = None


def load_mlx(model_path):
    global _MODEL, _PROC, _TOK
    from mlx_vlm.utils import load
    print(f"[mlx] chargement {model_path} ...", flush=True)
    _MODEL, _PROC = load(model_path)
    _TOK = _PROC.tokenizer if hasattr(_PROC, "tokenizer") else _PROC
    from mlx_vlm.generate.diffusion import is_diffusion_model
    if not is_diffusion_model(_MODEL):
        raise SystemExit(
            f"{model_path} n'est pas un modèle block-diffusion (pas de canvas_length). "
            "Ce viz attend DiffusionGemma.")
    print(f"[mlx] prêt. canvas={_MODEL.config.canvas_length} "
          f"mask_id={getattr(_TOK,'mask_token_id',None)}", flush=True)


def _build_prompt_ids(tok, prompt):
    import mlx.core as mx
    messages = [{"role": "user", "content": prompt}]
    ids = None
    if hasattr(tok, "apply_chat_template"):
        try:
            ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True)
        except Exception:
            ids = None
    if ids is None:
        ids = tok.encode(prompt)
    if isinstance(ids, dict) or hasattr(ids, "keys"):   # BatchEncoding
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while isinstance(ids, (list, tuple)) and len(ids) == 1 and \
            isinstance(ids[0], (list, tuple)):           # déballe [[...]]
        ids = ids[0]
    ids = [int(t) for t in ids]
    return mx.array([ids])


def mlx_diffusion_steps(prompt, max_steps=48, n_blocks=2, seed=None):
    """Générateur qui yield le même contrat de dict que le mock, depuis le vrai
    sampler entropy-bound de DiffusionGemma."""
    import mlx.core as mx
    from mlx_vlm.generate.common import generation_stream, wired_limit
    from mlx_vlm.generate import diffusion as D

    model, tok = _MODEL, _TOK
    if seed is not None:
        mx.random.seed(int(seed))

    input_ids = _build_prompt_ids(tok, prompt)
    dtype = input_ids.dtype

    cfg = model.config
    canvas_length = int(cfg.canvas_length)            # 256, on travaille en full canvas
    vocab = int(cfg.text_config.vocab_size)
    gen_cfg = D._diffusion_config_dict(getattr(cfg, "generation_config", None))
    sampler_cfg = D._diffusion_config_dict(gen_cfg.get("sampler_config"))
    entropy_bound = float(sampler_cfg.get("entropy_bound", 0.1))
    temp_cfg = {"t_min": float(gen_cfg.get("t_min", 0.4)),
                "t_max": float(gen_cfg.get("t_max", 0.8))}
    stopping_cfg = {"confidence_threshold": float(gen_cfg.get("confidence_threshold", 0.005)),
                    "stability_threshold": int(gen_cfg.get("stability_threshold", 1))}
    mask_id = getattr(tok, "mask_token_id", 4)
    eos_ids = set(gen_cfg.get("eos_token_id") or [])
    skip_ids = set(getattr(tok, "all_special_ids", []) or [])
    embed_scale = model.model.decoder.embed_scale

    def decode_clean(ids):
        try:
            return tok.decode([t for t in ids if t not in skip_ids])
        except Exception:
            return ""

    with mx.stream(generation_stream), wired_limit(model, [generation_stream]):
        soft_w = D._diffusion_soft_embedding_weight(model.model.decoder.embed_tokens)

        kv_cache = model.make_cache()
        # prefill du prompt (encoder causal) -> KV cache
        _, kv_cache = model.model.encoder(input_ids, attention_mask=None, cache=kv_cache)
        mx.eval([c.state for c in kv_cache])

        committed_disp = []   # tokens (filtrés) pour le texte affiché
        committed_count = 0
        step_global = 0
        stopped = False

        for block in range(n_blocks):
            current_canvas = D._diffusion_initialize_canvas(1, canvas_length, vocab, dtype)
            mask_mapping = model.model.decoder._make_decoder_masks(
                current_canvas[..., None], kv_cache, None)
            logits_no_sc, logits_sc = D._make_diffusion_decoder_logits_fns(
                model, kv_cache, mask_mapping, compile_graph=False)
            self_cond = None
            argmax_canvas = current_canvas
            prev_reveal = None
            history = []

            for cur_step in reversed(range(1, max_steps + 1)):
                # --- forward du décodeur diffusion ---
                if self_cond is None:
                    processed = logits_no_sc(current_canvas)
                else:
                    processed = logits_sc(current_canvas, self_cond)

                sched_t = D._diffusion_linear_temperature(cur_step, max_steps, temp_cfg)
                if sched_t is not None:
                    processed = processed / sched_t

                argmax_canvas = mx.argmax(processed, axis=-1).astype(dtype)
                denoiser_canvas = argmax_canvas  # temperature <= 0 (greedy)

                # --- masque d'acceptation entropy-bound (le coeur du sampler) ---
                if cur_step > 1:
                    token_entropy, next_sc = D._diffusion_entropy_and_soft_embeddings(
                        processed, soft_w, embed_scale)
                else:
                    token_entropy = D._diffusion_token_entropy(processed)
                    next_sc = None
                acceptance = D._diffusion_entropy_transfer_mask(token_entropy, entropy_bound)
                accepted = mx.where(acceptance, denoiser_canvas, current_canvas)
                current_canvas = mx.where(
                    acceptance, accepted,
                    D._diffusion_initialize_canvas(1, canvas_length, vocab, dtype))

                # confiance par position = proba top-1 du modèle
                conf = D._diffusion_token_probability(processed, argmax_canvas)

                mx.eval(argmax_canvas, acceptance, conf, token_entropy)

                draft_l = argmax_canvas[0].tolist()
                reveal_l = [bool(v) for v in acceptance[0].tolist()]
                conf_l = conf[0].tolist()
                ent_l = token_entropy[0].tolist()

                cells, n_masked, sum_h, rev_ids = [], 0, 0.0, []
                for i in range(CANVAS):
                    if i < canvas_length and reveal_l[i]:
                        tid = int(draft_l[i])
                        txt = tok.decode([tid]).replace("\n", "⏎").replace("\t", "⇥") or " "
                        cells.append(dict(t=txt[:5], s=S_STABLE, c=round(float(conf_l[i]), 3)))
                        if tid not in skip_ids:
                            rev_ids.append(tid)
                    elif i < canvas_length:
                        cells.append(dict(t=MASK, s=S_MASK, c=0.0))
                        n_masked += 1
                        sum_h += float(ent_l[i])
                    else:
                        cells.append(dict(t="", s=S_MASK, c=0.0))

                if prev_reveal is None:
                    newly, renoised = sum(reveal_l), 0
                else:
                    newly = sum(1 for i in range(canvas_length)
                                if reveal_l[i] and not prev_reveal[i])
                    renoised = sum(1 for i in range(canvas_length)
                                   if prev_reveal[i] and not reveal_l[i])
                prev_reveal = reveal_l

                text = decode_clean(committed_disp + rev_ids)
                yield dict(type="step", step=step_global, max_steps=max_steps, block=block,
                           cells=cells, newly=newly, renoised=renoised,
                           committed=committed_count,
                           mean_entropy=round(sum_h / max(1, n_masked), 3), text=text)
                step_global += 1

                # early-stop : canvas stable + confiant (comme la lib)
                if D._diffusion_stable_and_confident(argmax_canvas, processed,
                                                     history, stopping_cfg):
                    break
                # self-conditioning pour le prochain step
                if cur_step > 1:
                    if next_sc is None:
                        next_sc = D._diffusion_soft_embeddings(processed, soft_w, embed_scale)
                    self_cond = next_sc

            # --- commit du bloc : le canvas final = argmax ---
            current_canvas = argmax_canvas
            mx.eval(current_canvas)
            block_ids = [int(t) for t in current_canvas[0].tolist()]
            block_keep = []
            for tid in block_ids:
                if tid in eos_ids:
                    stopped = True
                    break
                block_keep.append(tid)
            committed_disp.extend(t for t in block_keep if t not in skip_ids)
            committed_count += len(block_keep)

            if stopped:
                break

            # avance le KV cache : on encode le bloc résolu (canvas complet)
            _, kv_cache = model.model.encoder(current_canvas, attention_mask=None, cache=kv_cache)
            mx.eval([c.state for c in kv_cache])
            mx.clear_cache()

        yield dict(type="done", text=decode_clean(committed_disp))


# ============================================================================
# SERVEUR
# ============================================================================
BACKEND = mock_diffusion_steps
DEFAULT_STEPS = 48
DEFAULT_BLOCKS = 2


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send_html()
        elif u.path == "/stream":
            self._send_stream(parse_qs(u.query))
        else:
            self.send_response(404)
            self.end_headers()

    def _send_html(self):
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self, q):
        if "probe" in q:
            self.send_response(204)
            self.end_headers()
            return
        prompt = unquote(q.get("prompt", ["Explique la diffusion de texte en 3 phrases."])[0])
        steps = int(q.get("steps", [DEFAULT_STEPS])[0])
        blocks = int(q.get("blocks", [DEFAULT_BLOCKS])[0])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for ev in BACKEND(prompt, max_steps=steps, n_blocks=blocks):
                payload = "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            err = json.dumps(dict(type="error", message=str(e)), ensure_ascii=False)
            try:
                self.wfile.write(("data: " + err + "\n\n").encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass


# ============================================================================
# UI EMBARQUÉE — esthétique instrument / persistance CRT phosphore.
# ============================================================================
HTML = r"""<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DiffusionGemma · denoise scope</title>
<style>
  :root{
    --bg:#070a0f; --panel:#0c111a;
    --line:#16202e; --mask:#141c28; --phos:#54e08a; --phos-dim:#2f6f4d;
    --cold:#3a6ea5; --warm:#d98a3d; --hot:#e0533a; --ink:#9fb2c4; --ink-dim:#5a6b7d;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:13px/1.4 ui-monospace,"JetBrains Mono","SF Mono",Menlo,monospace;}
  .wrap{max-width:1180px;margin:0 auto;padding:22px 20px 60px}
  header{display:flex;align-items:baseline;gap:14px;border-bottom:1px solid var(--line);
    padding-bottom:12px;margin-bottom:18px}
  header h1{font-size:15px;font-weight:600;letter-spacing:.5px;color:#cfe;margin:0}
  header .sub{color:var(--ink-dim);font-size:11px;letter-spacing:.8px;text-transform:uppercase}
  .bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:16px}
  input[type=text]{flex:1;min-width:280px;background:var(--panel);border:1px solid var(--line);
    color:#dfe;padding:9px 11px;border-radius:6px;font:inherit}
  input[type=text]:focus{outline:none;border-color:var(--phos-dim)}
  .num{width:62px;background:var(--panel);border:1px solid var(--line);color:#dfe;
    padding:9px 8px;border-radius:6px;font:inherit}
  label.k{color:var(--ink-dim);font-size:11px;letter-spacing:.5px}
  button{background:#10301f;border:1px solid var(--phos-dim);color:var(--phos);
    padding:9px 16px;border-radius:6px;font:inherit;cursor:pointer;letter-spacing:.5px}
  button:hover{background:#15402a}
  button.stop{background:#301414;border-color:#7a3030;color:#e0846a}
  .main{display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:20px}
  @media(max-width:860px){.main{grid-template-columns:1fr}}
  .scope{background:radial-gradient(120% 120% at 50% 0%,#0a1019,#060a0f);
    border:1px solid var(--line);border-radius:10px;padding:14px}
  #canvas{display:grid;grid-template-columns:repeat(16,1fr);gap:3px}
  .cell{aspect-ratio:1/1;border-radius:3px;background:var(--mask);
    display:flex;align-items:center;justify-content:center;overflow:hidden;
    font-size:9px;color:#02160c;font-weight:600;text-align:center;line-height:1;
    transition:background .18s ease, box-shadow .18s ease;position:relative}
  .cell span{padding:1px;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .cell.empty{background:#0a0f16;box-shadow:none}
  .cell.flash{box-shadow:0 0 0 1px var(--phos),0 0 10px var(--phos)}
  .cell.renoise{box-shadow:0 0 0 1px var(--hot),0 0 10px var(--hot)}
  .side{display:flex;flex-direction:column;gap:14px}
  .stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
  .stat h3{margin:0 0 8px;font-size:10px;letter-spacing:1.2px;text-transform:uppercase;color:var(--ink-dim)}
  .row{display:flex;justify-content:space-between;margin:3px 0}
  .row b{color:#dfe;font-weight:600}
  .big{font-size:26px;color:var(--phos);font-weight:600;letter-spacing:1px}
  #trace{width:100%;height:64px;display:block;background:#070b11;border:1px solid var(--line);border-radius:6px}
  .legend{display:flex;gap:12px;flex-wrap:wrap;font-size:10px;color:var(--ink-dim);margin-top:10px}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:-1px;margin-right:4px}
  .out{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;
    margin-top:18px;white-space:pre-wrap;color:#cdd;min-height:58px;font-size:12.5px}
  .out .cur{color:var(--phos)}
  .err{color:var(--hot);white-space:pre-wrap}
  .pill{font-size:10px;color:var(--ink-dim);border:1px solid var(--line);border-radius:20px;padding:2px 9px}
</style></head>
<body><div class="wrap">
  <header>
    <h1>denoise·scope</h1>
    <span class="sub">DiffusionGemma 26B-A4B · 256-token canvas</span>
    <span class="pill" id="mode">—</span>
  </header>

  <div class="bar">
    <input id="prompt" type="text" value="Écris une fonction Python qui inverse une chaîne, avec docstring.">
    <label class="k">steps</label><input id="steps" class="num" type="number" value="48" min="1" max="256">
    <label class="k">blocs</label><input id="blocks" class="num" type="number" value="2" min="1" max="8">
    <button id="run">▶ run</button>
    <button id="stop" class="stop" style="display:none">■ stop</button>
  </div>

  <div class="main">
    <div class="scope">
      <div id="canvas"></div>
      <div class="legend">
        <span><i style="background:#141c28"></i>masque</span>
        <span><i style="background:#3a6ea5"></i>conf. basse</span>
        <span><i style="background:#d98a3d"></i>conf. moyenne</span>
        <span><i style="background:#54e08a"></i>conf. haute</span>
        <span><i style="box-shadow:0 0 0 1px #e0533a;background:#1a0e0c"></i>re-bruité</span>
      </div>
    </div>

    <div class="side">
      <div class="stat">
        <h3>step</h3>
        <div class="big"><span id="step">0</span><span style="font-size:14px;color:var(--ink-dim)">/<span id="maxstep">48</span></span></div>
        <div class="row"><span>bloc</span><b id="block">0</b></div>
        <div class="row"><span>résolus / step</span><b id="newly">0</b></div>
        <div class="row"><span>re-bruités / step</span><b id="renoised">0</b></div>
      </div>
      <div class="stat">
        <h3>convergence</h3>
        <div class="row"><span>masqués restants</span><b id="masked">256</b></div>
        <div class="row"><span>entropie moy.</span><b id="ent">—</b></div>
        <div class="row"><span>commit total</span><b id="commit">0</b></div>
      </div>
      <div class="stat">
        <h3>trace · résolus(vert) / re-bruités(rouge)</h3>
        <canvas id="trace" width="252" height="64"></canvas>
      </div>
    </div>
  </div>

  <div class="out" id="out"><span style="color:var(--ink-dim)">la sortie décodée s'accumule ici…</span></div>
</div>

<script>
const $=id=>document.getElementById(id);
const canvasEl=$('canvas'); let cells=[];
for(let i=0;i<256;i++){const d=document.createElement('div');d.className='cell empty';
  d.innerHTML='<span></span>';canvasEl.appendChild(d);cells.push(d);}

const trace=$('trace'),tx=trace.getContext('2d'); let hist=[];
function drawTrace(){
  tx.clearRect(0,0,trace.width,trace.height);
  const n=hist.length||1, w=trace.width/Math.max(n,40), mid=trace.height-2;
  const mx=Math.max(8,...hist.map(h=>Math.max(h.n,h.r)));
  tx.strokeStyle='#16202e';tx.beginPath();tx.moveTo(0,mid);tx.lineTo(trace.width,mid);tx.stroke();
  hist.forEach((h,i)=>{
    const x=i*w;
    tx.fillStyle='#54e08a';const hn=(h.n/mx)*(trace.height-6);tx.fillRect(x,mid-hn,Math.max(1,w-1),hn);
    tx.fillStyle='#e0533a';const hr=(h.r/mx)*(trace.height-6);tx.fillRect(x,mid,Math.max(1,w-1),hr);
  });
}
function colorFor(c){
  if(c<=0)return '#141c28';
  if(c<0.4){const t=c/0.4;return mix([58,110,165],[217,138,61],t);}
  const t=(c-0.4)/0.6;return mix([217,138,61],[84,224,138],t);
}
function mix(a,b,t){return `rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')})`;}

let es=null, prevState=new Array(256).fill(0);
function reset(){hist=[];drawTrace();prevState.fill(0);
  cells.forEach(c=>{c.className='cell empty';c.firstChild.textContent='';});
  $('out').innerHTML='';}

function start(){
  if(es)es.close();
  reset();
  const p=encodeURIComponent($('prompt').value);
  const s=$('steps').value,b=$('blocks').value;
  es=new EventSource(`/stream?prompt=${p}&steps=${s}&blocks=${b}`);
  $('run').style.display='none';$('stop').style.display='';
  es.onmessage=e=>{
    const m=JSON.parse(e.data);
    if(m.type==='error'){$('out').innerHTML='<span class="err">⚠ '+m.message+'</span>';finish();return;}
    if(m.type==='done'){$('out').innerHTML=escapeHtml(m.text);finish();return;}
    render(m);
  };
  es.onerror=()=>{finish();};
}
function finish(){if(es){es.close();es=null;}$('run').style.display='';$('stop').style.display='none';}

function render(m){
  $('step').textContent=m.step+1; $('maxstep').textContent=m.max_steps;
  $('block').textContent=m.block; $('newly').textContent=m.newly;
  $('renoised').textContent=m.renoised; $('ent').textContent=m.mean_entropy;
  $('commit').textContent=m.committed;
  let masked=0;
  m.cells.forEach((c,i)=>{
    const el=cells[i];
    if(c.t===''){el.className='cell empty';el.firstChild.textContent='';prevState[i]=0;return;}
    if(c.s===0){
      masked++;
      el.className='cell';el.style.background='#141c28';el.firstChild.textContent='';
      if(prevState[i]===2){el.classList.add('renoise');}
      prevState[i]=0;
    }else{
      el.className='cell';el.style.background=colorFor(c.c);
      el.firstChild.textContent=c.t.length>5?c.t.slice(0,5):c.t;
      el.title=`${c.t}  conf=${c.c}`;
      if(prevState[i]!==2){el.classList.add('flash');}
      prevState[i]=2;
    }
  });
  $('masked').textContent=masked;
  hist.push({n:m.newly,r:m.renoised}); if(hist.length>120)hist.shift(); drawTrace();
  $('out').innerHTML='<span class="cur">'+escapeHtml(m.text)+'</span>';
}
function escapeHtml(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

$('run').onclick=start; $('stop').onclick=finish;
$('prompt').addEventListener('keydown',e=>{if(e.key==='Enter')start();});
$('mode').textContent=window.__MODE__||'mock';
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="UI seule, dynamique simulée")
    ap.add_argument("--model", default=None, help="chemin/repo HF du modèle MLX")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=2)
    args = ap.parse_args()

    global BACKEND, DEFAULT_STEPS, DEFAULT_BLOCKS, HTML
    DEFAULT_STEPS, DEFAULT_BLOCKS = args.steps, args.blocks

    if args.model and not args.mock:
        load_mlx(args.model)
        BACKEND = mlx_diffusion_steps
        HTML = HTML.replace("window.__MODE__||'mock'", f"'{args.model.split('/')[-1]}'")
    else:
        BACKEND = mock_diffusion_steps
        HTML = HTML.replace("window.__MODE__||'mock'", "'mock'")
        print("[mode] MOCK — lance avec --model <repo> pour le vrai sampler.", flush=True)

    # MLX réel : le sampler doit tourner dans le thread où le generation_stream
    # de la lib a été créé (le thread principal) -> serveur mono-thread. En mock
    # on garde le multi-thread pour une UI réactive.
    use_real = bool(args.model and not args.mock)
    server_cls = HTTPServer if use_real else ThreadingHTTPServer
    srv = server_cls(("127.0.0.1", args.port), Handler)
    print(f"[ok] http://127.0.0.1:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
