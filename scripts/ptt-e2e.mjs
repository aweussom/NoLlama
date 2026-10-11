// End-to-end check of the web UI's push-to-talk with a fake microphone.
//
//     node scripts/ptt-e2e.mjs --url http://localhost:8000 --wav jfk.wav [--hold-ms 12000] [--lang en]
//
// Why: the real path is MediaRecorder -> WebM/Opus -> browser decode ->
// 16 kHz WAV -> /v1/audio/transcriptions -> input box, and none of it runs
// in a unit test. Chrome can stand a WAV file in for the microphone
// (--use-file-for-fake-audio-capture), so the whole chain runs headless with
// no person speaking. What it cannot prove: a real microphone's levels and
// noise, permission prompts (auto-accepted here), or Firefox/Safari.
//
// Needs a running NoLlama with --whisper-dir. Node 22+ (global WebSocket),
// no npm packages. Prints JSON: whether the button showed, its tooltip, the
// transcript that landed in the input box. Exit 0 when the input box got
// non-empty text, 1 otherwise.
import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) =>
    a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc, []));
const url = args.url || 'http://localhost:8000';
const wav = resolve(args.wav || 'jfk.wav');
const holdMs = Number(args['hold-ms'] || 12000);
const lang = args.lang ?? 'en';  // the picker value to set; must be one of PTT_LANGS
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = args.chrome || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

/** Sleep for ms milliseconds. */
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/**
 * Minimal CDP client over the page's WebSocket.
 *
 * Why: no puppeteer, so the script runs on any box with node and Chrome.
 *
 * In: page WebSocket URL. Out: { send(method, params) -> result, close() }.
 * A CDP error rejects with its message.
 */
async function cdp(wsUrl) {
    const ws = new WebSocket(wsUrl);
    await new Promise((ok, fail) => { ws.onopen = ok; ws.onerror = fail; });
    let id = 0;
    const pending = new Map();
    ws.onmessage = (m) => {
        const msg = JSON.parse(m.data);
        if (msg.id && pending.has(msg.id)) {
            const { ok, fail } = pending.get(msg.id);
            pending.delete(msg.id);
            msg.error ? fail(new Error(msg.error.message)) : ok(msg.result);
        }
    };
    return {
        send: (method, params = {}) => new Promise((ok, fail) => {
            pending.set(++id, { ok, fail });
            ws.send(JSON.stringify({ id, method, params }));
        }),
        close: () => ws.close(),
    };
}

/**
 * Evaluate an expression in the page and return its value.
 *
 * Why: every check here is "ask the page", so one helper keeps the
 * awaitPromise/returnByValue boilerplate out of the steps.
 *
 * In: CDP client, JS expression. Out: the value; throws on a page exception.
 */
async function evaluate(c, expression) {
    const r = await c.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    return r.result.value;
}

const profile = mkdtempSync(join(tmpdir(), 'ptt-e2e-'));
const proc = spawn(chrome, [
    '--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    `--use-file-for-fake-audio-capture=${wav}`, '--autoplay-policy=no-user-gesture-required',
    url,
], { stdio: 'ignore' });

let code = 1;
const out = { url, wav, hold_ms: holdMs, lang };
try {
    let target;
    for (let i = 0; i < 50 && !target; i++) {
        await sleep(200);
        try {
            const list = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
            target = list.find(t => t.type === 'page');
        } catch {}
    }
    if (!target) throw new Error('Chrome did not expose a page');
    const c = await cdp(target.webSocketDebuggerUrl);
    await c.send('Runtime.enable');
    for (let i = 0; i < 50; i++) {
        await sleep(200);
        // The target exists before the page has parsed; ask defensively.
        if (await evaluate(c, "(() => { const b = document.getElementById('ptt-btn'); return !!b && !b.hidden && document.readyState === 'complete'; })()")) break;
    }
    out.button_shown = await evaluate(c, "!!document.getElementById('ptt-btn') && !document.getElementById('ptt-btn').hidden");
    out.button_disabled = await evaluate(c, "document.getElementById('ptt-btn').disabled");
    out.title = await evaluate(c, "document.getElementById('ptt-btn').title");
    out.language = await evaluate(c, "document.getElementById('ptt-lang').value");
    if (out.button_shown && !out.button_disabled) {
        await evaluate(c, `document.getElementById('ptt-lang').value = ${JSON.stringify(lang)}`);
        await evaluate(c, 'pttStart()');
        await sleep(holdMs);
        out.recording_class = await evaluate(c, "document.getElementById('ptt-btn').classList.contains('recording')");
        const t0 = Date.now();
        await evaluate(c, 'pttStop()');
        for (let i = 0; i < 300; i++) {
            await sleep(200);
            const busy = await evaluate(c, "document.getElementById('ptt-btn').classList.contains('busy')");
            const val = await evaluate(c, 'input.value');
            if (!busy && (val || i > 10)) break;
        }
        out.release_to_text_ms = Date.now() - t0;
        out.transcript = await evaluate(c, 'input.value');
        out.placeholder = await evaluate(c, 'input.placeholder');
        code = out.transcript ? 0 : 1;
    }
    c.close();
} catch (e) {
    out.error = String(e && e.message || e);
} finally {
    proc.kill();
}
console.log(JSON.stringify(out, null, 2));
process.exit(code);
