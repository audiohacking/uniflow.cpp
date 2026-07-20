// UniFlow WebGPU demo — Small pack, DiT Q8_0, T5 Q4_0 (JS tokenizer).
const HF =
  "https://huggingface.co/audiohacking/uniflow-audio-gguf/resolve/main/uniflow-audio-v1.1-small";

const FILES = {
  t5: `${HF}/t5_encoder.gguf`,
  dit: `${HF}/dit-Q8_0.gguf`,
  vae: `${HF}/vae.gguf`,
  instructions: `${HF}/instructions.gguf`,
};

const logEl = document.getElementById("log");
const btnLoad = document.getElementById("btn-load");
const btnGen = document.getElementById("btn-gen");
const player = document.getElementById("player");

function log(msg) {
  logEl.textContent += `${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
  console.log(msg);
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

async function fetchToMemfs(mod, url, vfsPath) {
  log(`fetch ${url.split("/").pop()} …`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const buf = new Uint8Array(await res.arrayBuffer());
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
  mod.FS.writeFile(vfsPath, buf);
  log(`  → ${vfsPath} (${(buf.byteLength / 1e6).toFixed(1)} MB)`);
}

async function loadTokenizer() {
  // Lightweight T5 Unigram via transformers.js (CDN)
  const { AutoTokenizer } = await import(
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0"
  );
  log("loading JS T5 tokenizer…");
  const tok = await AutoTokenizer.from_pretrained("Xenova/t5-small", {
    // vocab compatible enough for demo; prefer flan-t5-large when mirrored
    progress_callback: () => {},
  });
  // Better: use google/flan-t5-large if transformers can load spiece remotely
  try {
    const flan = await AutoTokenizer.from_pretrained("Xenova/t5-base");
    log("tokenizer ready (Xenova/t5-base proxy)");
    return flan;
  } catch (e) {
    log(`flan fallback: ${e}`);
    return tok;
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
    const backend = mod.initBackend();
    log(`backend: ${backend}`);
    if (String(backend).startsWith("error:")) {
      throw new Error(backend);
    }

    await fetchToMemfs(mod, FILES.t5, "/models/t5_encoder.gguf");
    await fetchToMemfs(mod, FILES.dit, "/models/dit-Q8_0.gguf");
    await fetchToMemfs(mod, FILES.vae, "/models/vae.gguf");
    await fetchToMemfs(mod, FILES.instructions, "/models/instructions.gguf");

    const ok = mod.loadModels(
      "/models/t5_encoder.gguf",
      "/models/dit-Q8_0.gguf",
      "/models/vae.gguf",
      "/models/instructions.gguf"
    );
    if (!ok) throw new Error(mod.lastError() || "loadModels failed");
    log("models loaded");

    tokenizer = await loadTokenizer();
    btnGen.disabled = false;
    log("ready — enter a caption and Generate");
  } catch (e) {
    log(`ERROR: ${e.message || e}`);
    btnLoad.disabled = false;
  }
};

btnGen.onclick = async () => {
  btnGen.disabled = true;
  try {
    const mod = await wasmReady();
    const caption = document.getElementById("caption").value.trim();
    const duration = Number(document.getElementById("duration").value);
    const steps = Number(document.getElementById("steps").value);
    const seed = Number(document.getElementById("seed").value);

    log(`tokenize: "${caption}"`);
    const encoded = await tokenizer(caption, {
      add_special_tokens: true,
      truncation: true,
      max_length: 128,
    });
    // transformers.js returns { input_ids: Tensor|number[] }
    let ids = encoded.input_ids;
    if (ids && ids.data) ids = Array.from(ids.data);
    else if (ids && ids.tolist) ids = ids.tolist();
    else ids = Array.from(ids);
    // Ensure EOS (1) for T5
    if (ids.length === 0 || ids[ids.length - 1] !== 1) ids.push(1);
    log(`tokens=${ids.length}`);

    log("generating (this can take a while)…");
    const n = mod.generateFromTokens(ids, duration, steps, 5.0, -1.0, seed);
    if (!n) throw new Error(mod.lastError() || "generate failed");

    const ptr = mod.pcmPointer();
    const rate = mod.pcmSampleRate();
    const samples = new Float32Array(mod.HEAPF32.buffer, ptr, n).slice();
    mod.freePcm();

    const ctx = new AudioContext({ sampleRate: rate });
    const buf = ctx.createBuffer(1, samples.length, rate);
    buf.copyToChannel(samples, 0);
    const blob = await audioBufferToWavBlob(buf);
    player.src = URL.createObjectURL(blob);
    await player.play().catch(() => {});
    log(`done — ${samples.length} samples @ ${rate} Hz`);
  } catch (e) {
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

log("Click “Load models” (needs WebGPU + ~520 MB download).");
