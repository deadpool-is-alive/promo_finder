"""One tiny client for any chat model. Swap models by changing env vars, no code change.
 provider=openai    -> any OpenAI-compatible /chat/completions API (OpenAI, Gemini, Groq, OpenRouter, Ollama...)
 provider=anthropic -> Anthropic Messages API
Features: disk cache (re-runs are free), retry/backoff for free-tier rate limits, JSON repair, token accounting."""
from __future__ import annotations
import hashlib, json, os, re, threading, time
from pathlib import Path
import requests


def extract_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except Exception:
        pass
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = json.JSONDecoder().raw_decode(t[i:])
                return obj
            except Exception:
                continue
    raise ValueError("no JSON found in model reply")


class LLM:
    def __init__(self, provider="openai", base_url="", api_key="", fast_model="", strong_model="",
                 cache_dir=".cache", min_interval=0.0, json_mode=True, temperature="0.2", token_param="max_tokens"):
        self.provider, self.base_url, self.api_key = provider, base_url, api_key
        self.models = {"fast": fast_model, "strong": strong_model or fast_model}
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self.min_interval, self.json_mode, self.temperature, self.token_param = min_interval, json_mode, temperature, token_param
        self.lock, self.last_call = threading.Lock(), 0.0
        self.usage = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "failures": 0}
        self.by_label: dict = {}

    @classmethod
    def from_env(cls, **kw):
        e = os.environ.get
        prov = e("LLM_PROVIDER", "openai")
        d_fast = {"anthropic": "claude-haiku-5-5"}.get(prov, "")
        d_strong = {"anthropic": "claude-sonnet-5-5"}.get(prov, "")
        fast = e("LLM_MODEL_FAST", d_fast)
        d_url = "https://api.anthropic.com/v1" if prov == "anthropic" else "https://api.openai.com/v1"
        return cls(prov, e("LLM_BASE_URL", d_url), e("LLM_API_KEY", ""), fast,
                   e("LLM_MODEL_STRONG", d_strong) or fast, e("LLM_CACHE_DIR", ".cache"),
                   float(e("LLM_MIN_INTERVAL", "0")), e("LLM_JSON_MODE", "1") == "1",
                   e("LLM_TEMPERATURE", "0.2"), e("LLM_TOKEN_PARAM", "max_tokens"), **kw)

    # ---- raw HTTP -------------------------------------------------------
    def _throttle(self):
        if self.min_interval <= 0:
            return
        with self.lock:
            wait = self.last_call + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self.last_call = time.time()

    def _http(self, model, system, user, max_tokens):
        if self.provider == "anthropic":
            r = requests.post(self.base_url.rstrip("/") + "/messages",
                              headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                              json={"model": model, "max_tokens": max_tokens, "system": system,
                                    "temperature": float(self.temperature or 0.2),
                                    "messages": [{"role": "user", "content": user}]}, timeout=180)
        else:
            body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    self.token_param: max_tokens}
            if self.temperature != "":
                body["temperature"] = float(self.temperature)
            if self.json_mode:
                body["response_format"] = {"type": "json_object"}
            r = requests.post(self.base_url.rstrip("/") + "/chat/completions",
                              headers={"Authorization": f"Bearer {self.api_key}"}, json=body, timeout=180)
        return r

    def _text_and_usage(self, r):
        j = r.json()
        if self.provider == "anthropic":
            text = "".join(b.get("text", "") for b in j.get("content", []))
            u = j.get("usage", {})
            return text, u.get("input_tokens", 0), u.get("output_tokens", 0)
        text = j["choices"][0]["message"]["content"] or ""
        u = j.get("usage", {})
        return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

    def _request(self, model, system, user, max_tokens):
        err = None
        for attempt in range(6):
            self._throttle()
            try:
                r = self._http(model, system, user, max_tokens)
            except requests.RequestException as ex:
                err = str(ex); time.sleep(2 ** attempt); continue
            if r.status_code == 200:
                return self._text_and_usage(r)
            if r.status_code == 400 and self.provider != "anthropic" and self.json_mode:
                self.json_mode = False          # endpoint doesn't support JSON mode; retry without it
                continue
            err = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code in (408, 409, 425, 429, 500, 502, 503, 504, 529):
                ra = r.headers.get("retry-after")
                time.sleep(min(60, float(ra) if ra and ra.replace('.', '').isdigit() else 2 ** (attempt + 1)))
                continue
            break
        raise RuntimeError(err or "request failed")

    # ---- public ---------------------------------------------------------
    def call(self, system: str, user: str, tier="fast", max_tokens=6000, label="call"):
        model = self.models[tier]
        key = hashlib.sha256(json.dumps([self.provider, model, system, user]).encode()).hexdigest()
        f = self.cache / f"{key}.json"
        if f.exists():
            with self.lock:
                self.usage["cache_hits"] += 1
            return json.loads(f.read_text())["data"]
        text, tin, tout = self._request(model, system, user, max_tokens)
        try:
            data = extract_json(text)
        except ValueError:
            t2, i2, o2 = self._request(model, system, user + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object.", max_tokens)
            tin, tout = tin + i2, tout + o2
            data = extract_json(t2)
        with self.lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += tin
            self.usage["output_tokens"] += tout
            b = self.by_label.setdefault(label, {"calls": 0, "in": 0, "out": 0, "model": model})
            b["calls"] += 1; b["in"] += tin; b["out"] += tout
        f.write_text(json.dumps({"model": model, "data": data}))
        return data
