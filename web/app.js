// UniFlow WebGPU demo — Small pack, DiT Q4_0 + T5 Q4_0 (JS tokenizer).
const HF =
  "https://huggingface.co/audiohacking/uniflow-audio-gguf/resolve/main/uniflow-audio-v1.1-small";

const FILES = [
  { key: "t5", url: `${HF}/t5_encoder.gguf`, path: "/models/t5_encoder.gguf", approxMB: 183 },
  { key: "dit", url: `${HF}/dit-Q4_0.gguf`, path: "/models/dit-Q4_0.gguf", approxMB: 170 },
  { key: "vae", url: `${HF}/vae.gguf`, path: "/models/vae.gguf", approxMB: 88 },
  { key: "instructions", url: `${HF}/instructions.gguf`, path: "/models/instructions.gguf", approxMB: 9 },
];

const logEl = document.getElementById("log");
const btnLoad = document.getElementById("btn-load");
const btnGen = document.getElementById("btn-gen");
const player = document.getElementById("player");
const progressWrap = document.getElementById("progress-wrap");
const progressBar = document.getElementById("progress-bar");
const progressLabel = document.getElementById("progress-label");

function log(msg) {
  logEl.textContent += `${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
  console.log(msg);
}

function toNum(v) {
  if (typeof v === "bigint") return Number(v);
  return Number(v);
}

function setProgress(pct, label) {
  progressWrap.hidden = false;
  const p = Math.max(0, Math.min(100, pct));
  progressBar.style.width = `${p.toFixed(1)}%`;
  progressLabel.textContent = label || `${p.toFixed(0)}%`;
}

function hideProgress() {
  progressWrap.hidden = true;
  progressBar.style.width = "0%";
  progressLabel.textContent = "";
}

function wasmReady() {
  return new Promise((resolve) => {
    if (typeof Module !== "undefined" && Module.calledRun) {
      resolve(Module);
      return;
    }
    window.Module = window.Module || {};
    const prev = window.Module.onRuntimeInitialized;
    window.Module.onRuntimeInitialized = () => {
      if (prev) prev();
      resolve(Module);
    };
  });
}

function ensureDir(mod, vfsPath) {
  const parts = vfsPath.split("/").filter(Boolean);
  let dir = "";
  for (let i = 0; i < parts.length - 1; i++) {
    dir += "/" + parts[i];
    try {
      mod.FS.mkdir(dir);
    } catch (_) {
      /* exists */
    }
  }
}

async function fetchToMemfs(mod, file, onFileProgress) {
  const name = file.url.split("/").pop();
  log(`fetch ${name} …`);
  const res = await fetch(file.url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${file.url}`);

  const totalHdr = Number(res.headers.get("Content-Length") || 0);
  const total = totalHdr > 0 ? totalHdr : Math.round(file.approxMB * 1e6);
  const reader = res.body && res.body.getReader ? res.body.getReader() : null;

  let buf;
  if (!reader) {
    buf = new Uint8Array(await res.arrayBuffer());
    onFileProgress(buf.byteLength, buf.byteLength, name);
  } else {
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      onFileProgress(received, total, name);
    }
    buf = new Uint8Array(received);
    let offset = 0;
    for (const c of chunks) {
      buf.set(c, offset);
      offset += c.byteLength;
    }
  }

  ensureDir(mod, file.path);
  mod.FS.writeFile(file.path, buf);
  log(`  → ${file.path} (${(buf.byteLength / 1e6).toFixed(1)} MB)`);
  return buf.byteLength;
}

async function loadTokenizer() {
  const { AutoTokenizer } = await import(
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0"
  );
  log("loading JS T5 tokenizer…");
  try {
    const flan = await AutoTokenizer.from_pretrained("Xenova/t5-base");
    log("tokenizer ready (Xenova/t5-base proxy)");
    return flan;
  } catch (e) {
    log(`t5-base failed (${e}); falling back to t5-small`);
    return AutoTokenizer.from_pretrained("Xenova/t5-small");
  }
}

let tokenizer = null;

btnLoad.onclick = async () => {
  btnLoad.disabled = true;
  try {
    if (!navigator.gpu) {
      throw new Error("WebGPU not available in this browser");
    }
    const mod = await wasmReady();
    log(`wasm ready`);
    setProgress(0, "init WebGPU…");
    // ASYNCIFY: WebGPU WaitAny may suspend → embind returns a Promise.
    const backend = await Promise.resolve(mod.initBackend());
    log(`backend: ${backend}`);
    if (String(backend).startsWith("error:")) {
      throw new Error(backend);
    }

    const approxTotal = FILES.reduce((s, f) => s + f.approxMB * 1e6, 0);
    let downloaded = 0;
    for (const file of FILES) {
      const before = downloaded;
      const nbytes = await fetchToMemfs(mod, file, (received, total, name) => {
        const fileFrac = total > 0 ? Math.min(1, received / total) : 0;
        const overall = before + fileFrac * (total || file.approxMB * 1e6);
        const pct = (100 * overall) / approxTotal;
        setProgress(
          pct,
          `Downloading ${name} — ${Math.min(100, (100 * received) / (total || 1)).toFixed(0)}% · overall ${pct.toFixed(0)}%`
        );
      });
      downloaded += nbytes;
      setProgress((100 * downloaded) / approxTotal, `Downloaded ${file.path.split("/").pop()}`);
    }

    setProgress(100, "Loading models into WASM…");
    const ok = await Promise.resolve(
      mod.loadModels(
        "/models/t5_encoder.gguf",
        "/models/dit-Q4_0.gguf",
        "/models/vae.gguf",
        "/models/instructions.gguf"
      )
    );
    if (!ok) throw new Error((await Promise.resolve(mod.lastError())) || "loadModels failed");
    log("models loaded (Small DiT Q4_0 + T5 Q4_0)");

    setProgress(100, "Loading tokenizer…");
    tokenizer = await loadTokenizer();
    btnGen.disabled = false;
    hideProgress();
    log("ready — enter a caption and Generate");
  } catch (e) {
    hideProgress();
    log(`ERROR: ${e.message || e}`);
    btnLoad.disabled = false;
  }
};

btnGen.onclick = async () => {
  btnGen.disabled = true;
  try {
    const mod = await wasmReady();
    const caption = document.getElementById("caption").value.trim();
    const duration = toNum(document.getElementById("duration").value);
    const steps = toNum(document.getElementById("steps").value) | 0;
    const seed = toNum(document.getElementById("seed").value) | 0;

    log(`tokenize: "${caption}"`);
    setProgress(0, "Tokenizing…");
    const encoded = await tokenizer(caption, {
      add_special_tokens: true,
      truncation: true,
      max_length: 128,
    });
    // transformers.js often returns BigInt token ids — coerce for embind.
    let ids = encoded.input_ids;
    if (ids && ids.data) ids = Array.from(ids.data);
    else if (ids && ids.tolist) ids = ids.tolist();
    else ids = Array.from(ids);
    ids = ids.map((x) => toNum(x) | 0);
    if (ids.length === 0 || ids[ids.length - 1] !== 1) ids.push(1);
    log(`tokens=${ids.length}`);

    setProgress(15, "Generating on WebGPU (can take a bit)…");
    log("generating (this can take a while)…");
    const nRaw = await Promise.resolve(
      mod.generateFromTokens(ids, duration, steps, 5.0, -1.0, seed)
    );
    const n = toNum(nRaw) | 0;
    if (!n) {
      throw new Error((await Promise.resolve(mod.lastError())) || "generate failed");
    }

    const pcm = await Promise.resolve(mod.getPcm());
    if (!pcm || pcm.length === 0) {
      throw new Error((await Promise.resolve(mod.lastError())) || "empty PCM");
    }
    const rate = toNum(await Promise.resolve(mod.pcmSampleRate())) | 0;
    const samples = pcm instanceof Float32Array ? pcm.slice() : Float32Array.from(pcm);
    await Promise.resolve(mod.freePcm());

    setProgress(90, "Encoding WAV…");
    const ctx = new AudioContext({ sampleRate: rate });
    const buf = ctx.createBuffer(1, samples.length, rate);
    buf.copyToChannel(samples, 0);
    const blob = await audioBufferToWavBlob(buf);
    player.src = URL.createObjectURL(blob);
    await player.play().catch(() => {});
    hideProgress();
    log(`done — ${samples.length} samples @ ${rate} Hz`);
  } catch (e) {
    hideProgress();
    log(`ERROR: ${e.message || e}`);
  } finally {
    btnGen.disabled = false;
  }
};

function audioBufferToWavBlob(buffer) {
  const numCh = 1;
  const sr = buffer.sampleRate;
  const data = buffer.getChannelData(0);
  const pcm = new Int16Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const s = Math.max(-1, Math.min(1, data[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  const ab = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(ab);
  const w = (o, s) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  w(0, "RIFF");
  v.setUint32(4, 36 + pcm.length * 2, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, numCh, true);
  v.setUint32(24, sr, true);
  v.setUint32(28, sr * numCh * 2, true);
  v.setUint16(32, numCh * 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, pcm.length * 2, true);
  new Int16Array(ab, 44).set(pcm);
  return new Blob([ab], { type: "audio/wav" });
}

log("Click “Load models” (WebGPU + ~450 MB download, Small Q4).");
