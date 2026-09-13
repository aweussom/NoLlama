// NoLlama Web UI

const chat = document.getElementById('chat');
const input = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const modelSelect = document.getElementById('model-select');
const statusDot = document.getElementById('status-dot');
const fileInput = document.getElementById('file-input');
const attachBtn = document.getElementById('attach-btn');
const imagePreview = document.getElementById('image-preview');
const previewImg = document.getElementById('preview-img');
const removeImageBtn = document.getElementById('remove-image');
const dropOverlay = document.getElementById('drop-overlay');
const newChatBtn = document.getElementById('new-chat-btn');
const temperatureSlider = document.getElementById('temperature');
const tempValue = document.getElementById('temp-value');
const noThinkCheckbox = document.getElementById('no-think');
// The last sentence is Muse Glimmer's native reasoning control (its chat
// template defers to a system-prompt 'Reasoning strength:' line, default
// high); other models read it as ordinary prose. Never mention '<think>'
// here: the model mimics the literal tags into its answer text (observed
// on Glimmer 2026-08-13).
const NO_THINK_PROMPT = 'Respond directly and concisely, with no internal reasoning preamble. Reasoning strength: minimal.';

// Temperature slider display
temperatureSlider.addEventListener('input', () => {
    tempValue.textContent = (temperatureSlider.value / 100).toFixed(1);
});

let chatHistory = [];
let attachedImage = null; // base64 data URI
let thinkExpanded = false; // track think block expand state across re-renders
let isGenerating = false;
let abortController = null;
// Last device that served a /chat/completions response (from X-Device header).
// The history-length banner is NPU-only: only NPU has a hard prompt-token cap.
let device = '';

/**
 * Flip the UI into/out of "generating" mode.
 *
 * Why: the send-button doubles as a visible Stop while generating (Escape
 * still works) — one button, two labels, no extra chrome.
 */
function setGenerating(on) {
    isGenerating = on;
    sendBtn.textContent = on ? 'Stop' : 'Send';
    sendBtn.classList.toggle('stop', on);
}

/**
 * Stop the current generation: abort our fetch AND tell the server.
 *
 * Why both: aborting the fetch only closes our connection — the server keeps
 * generating into the void (OpenVINO can't see the disconnect mid-stream), so
 * /v1/cancel asks it to stop via the streamer callback too.
 */
function cancelGeneration() {
    if (abortController) abortController.abort();
    fetch('/v1/cancel', { method: 'POST' }).catch(() => {});
}

/**
 * True when the user is within 80px of the chat bottom — i.e. following the
 * conversation rather than reading back. The margin absorbs sub-line scroll
 * jitter from streaming redraws.
 */
function shouldAutoScroll() {
    return chat.scrollHeight - chat.scrollTop - chat.clientHeight < 80;
}

/**
 * Scroll the chat to the bottom, but ONLY if the user was already there —
 * yanking the view down while someone reads an earlier answer is the classic
 * chat-UI sin.
 */
function scrollToBottom() {
    if (shouldAutoScroll()) chat.scrollTop = chat.scrollHeight;
}

// --- Sticky-bottom scroll for the live thinking block while streaming ---
// The DOM is wholesale re-rendered on every token (innerHTML = renderMarkdown),
// which normally destroys a scrollable .think-full and snaps its scroll to 0.
// streamState survives the redraw: `pinned` means "the user is watching the
// live tail", so each redraw re-pins to the bottom; a single scroll-up clears
// the flag, after which redraws preserve the user's relative view ("freed").
const STREAM_THRESHOLD = 32; // px from the bottom == "pinned to the stream"
let streamState = { thinkFull: null, pinned: true, onScroll: null };

/**
 * (Re)bind the scroll listener to the current .think-full node.
 *
 * Why rebinding exists: the node is recreated on every innerHTML redraw
 * during streaming — detach from the old one, bind to the new one. `pinned`
 * lives on streamState and is read at redraw time, not captured at the
 * listener's creation, so it persists correctly across redraws.
 */
function attachThinkScroll(thinkFull) {
    if (streamState.onScroll && streamState.thinkFull) {
        streamState.thinkFull.removeEventListener('scroll', streamState.onScroll);
    }
    streamState.thinkFull = thinkFull;
    streamState.onScroll = function () {
        const tf = streamState.thinkFull;
        if (!tf) return;
        streamState.pinned = tf.scrollHeight - tf.scrollTop - tf.clientHeight <= STREAM_THRESHOLD;
    };
    thinkFull.addEventListener('scroll', streamState.onScroll, { passive: true });
}

/**
 * Redraw the streaming assistant bubble for the accumulated text so far.
 *
 * Why the two paths: when a scrollable .think-full already exists, only its
 * inner content is swapped — the user's scrollTop survives natively, no
 * element recreation, no manual restore, no fighting the user's wheel.
 * Pinned re-pins to the tail; freed leaves the position untouched. Only when
 * no such node exists (or the think block just closed) is the bubble fully
 * re-rendered.
 */
function updateStreamBubble(assistantDiv, fullText) {
    const prev = streamState.thinkFull;
    if (prev) {
        const scratch = document.createElement('div');
        scratch.innerHTML = renderMarkdown(fullText, true);
        const newFull = scratch.querySelector('.think-full');
        if (newFull) {
            prev.innerHTML = newFull.innerHTML; // same element; scroll preserved
            const block = prev.closest('.think-block');
            const newBlock = scratch.querySelector('.think-block');
            if (block && newBlock) {
                const preview = block.querySelector('.think-preview');
                const newPreview = newBlock.querySelector('.think-preview');
                if (preview && newPreview) preview.innerHTML = newPreview.innerHTML;
                const header = block.querySelector('.think-header');
                const newHeader = newBlock.querySelector('.think-header');
                if (header && newHeader) header.innerHTML = newHeader.innerHTML;
                block.classList.toggle('collapsed', newBlock.classList.contains('collapsed'));
                block.classList.toggle('streaming', newBlock.classList.contains('streaming'));
                // Just-answer button lives after the think block; sync it below.
            }
            syncAnswerNodes(assistantDiv, scratch);
            if (streamState.pinned) prev.scrollTop = prev.scrollHeight;
        } else {
            // Think block closed — rebuild the whole bubble.
            assistantDiv.innerHTML = renderMarkdown(fullText, true);
            const tf = assistantDiv.querySelector('.think-full');
            if (tf) attachThinkScroll(tf);
        }
    } else {
        // No scrollable think block yet — normal full re-render, then attach
        // the scroll listener the first time a .think-full appears.
        assistantDiv.innerHTML = renderMarkdown(fullText, true);
        const thinkFull = assistantDiv.querySelector('.think-full');
        if (thinkFull) {
            attachThinkScroll(thinkFull);
            if (streamState.pinned) thinkFull.scrollTop = thinkFull.scrollHeight;
        }
    }
    scrollToBottom(); // keep the outer chat at the bottom when viewing the tail
}

/**
 * Keep the surviving .think-block, drop everything else in assistantDiv, then
 * re-append the answer nodes (and just-answer button) from the scratch render.
 */
function syncAnswerNodes(assistantDiv, scratch) {
    const keepBlock = assistantDiv.querySelector('.think-block');
    Array.from(assistantDiv.children).forEach((c) => { if (c !== keepBlock) c.remove(); });
    Array.from(scratch.children).forEach((c) => {
        if (!c.classList || !c.classList.contains('think-block')) assistantDiv.appendChild(c);
    });
}

/**
 * Detach the scroll listener and reset to "pinned" for the next stream —
 * called at stream start AND end so a leftover freed state from the previous
 * answer can't leave the new one unpinned.
 */
function resetStreamState() {
    if (streamState.onScroll && streamState.thinkFull) {
        streamState.thinkFull.removeEventListener('scroll', streamState.onScroll);
    }
    streamState.thinkFull = null;
    streamState.pinned = true;
    streamState.onScroll = null;
}

// --- Init ---

/**
 * Fold one SSE delta back into the <think>-tagged text the renderer expects.
 *
 * Why: the server streams reasoning as delta.reasoning_content (the field
 * OpenAI-compatible agent clients render as live thinking) and the answer as
 * delta.content; this UI's renderer keys on literal <think> tags, so the two
 * channels are re-tagged here rather than teaching the renderer a second input
 * shape. In: think = {open} carried across one stream, the delta object (may
 * be {} on keep-alive frames). Out: text to append to fullText, '' if nothing.
 */
function appendDelta(think, delta) {
    let out = '';
    const reasoning = delta.reasoning_content || delta.reasoning;
    if (reasoning) {
        if (!think.open) { out += '<think>'; think.open = true; }
        out += reasoning;
    }
    if (delta.content) out += closeThink(think) + delta.content;
    return out;
}

/** Close an open think block for appendDelta's state; '' when none is open. */
function closeThink(think) {
    if (!think.open) return '';
    think.open = false;
    return '</think>\n';
}

/** Page boot: health + model list, then poll health every 15s. */
async function init() {
    await checkHealth();
    await loadModels();
    setInterval(checkHealth, 15000);
    input.focus();
}

/**
 * Update the status dot from /health; a fetch failure shows as
 * "disconnected" rather than leaving a stale green dot lying.
 */
async function checkHealth() {
    try {
        const resp = await fetch('/health');
        const data = await resp.json();
        statusDot.className = 'status-dot ' + data.status;
        statusDot.title = data.status;
    } catch {
        statusDot.className = 'status-dot error';
        statusDot.title = 'disconnected';
    }
}

/**
 * Fill the model picker from /v1/models. Ids stay "name@DEVICE" (that is
 * what the server routes on); only the label is prettified to "name (DEVICE)".
 */
async function loadModels() {
    try {
        const resp = await fetch('/v1/models');
        const data = await resp.json();
        modelSelect.innerHTML = '';
        for (const m of data.data) {
            const opt = document.createElement('option');
            opt.value = m.id;  // e.g. "qwen3-8b-int4-cw@NPU"
            // Display as "qwen3-8b-int4-cw (NPU)"
            const parts = m.id.split('@');
            const name = parts[0];
            const device = parts[1] || m.owned_by.replace('local-', '').toUpperCase();
            opt.textContent = `${name} (${device})`;
            modelSelect.appendChild(opt);
        }
    } catch {}
}

// --- Request builder ---

/**
 * Assemble the /v1/chat/completions body from UI state.
 *
 * Why the UI-only repetition_penalty 1.1: the web UI is the "does it work"
 * surface, and a thinking-loop on a slow iGPU reads as "it doesn't"; the
 * server default stays milder (1.05) for coding agents. The no-think system
 * prompt replaces any user system message rather than stacking on it.
 */
function buildRequestBody(overrides) {
    const temp = temperatureSlider.value / 100;
    const noThink = noThinkCheckbox.checked;

    let messages = [...chatHistory];
    if (noThink) {
        // Prepend no-think system prompt
        messages = [
            { role: 'system', content: NO_THINK_PROMPT },
            ...messages.filter(m => m.role !== 'system'),
        ];
    }

    const body = {
        model: modelSelect.value,
        messages: messages,
        stream: true,
        max_tokens: 16384,
        // Matches the server default. This was 1.1 (Ollama's value) to break
        // thinking-loops faster on a slow iGPU, but 1.1 DEGENERATES the Phi-3
        // family: the penalty pushes an already-used vocabulary toward ever
        // rarer tokens and penalises EOS along with them, so the model can
        // never stop. [OBSERVED 2026-09-13, bare openvino_genai on CPU,
        // Phi-3.5-mini-int4-cw, greedy, max_new_tokens=1024] no penalty ->
        // 1989 chars, clean stop; 1.05 -> 3118 chars, clean stop; 1.1 -> 4722
        // chars ending "...urbane verdancy wrath workmanship Xanadu Yggdrasil
        // Zamians" and still going. The loop-breaking this bought is now done
        // properly by enable_thinking=false (_apply_thinking_switch), which
        // forecloses the channel instead of penalising a way out of it.
        repetition_penalty: 1.05,
    };

    if (temp > 0) {
        body.temperature = temp;
    }

    return { ...body, ...overrides };
}

// --- Just answer me, dammit! ---

/**
 * "Just answer me, dammit!": abort the thinking-heavy generation and re-ask
 * the same question with the no-think system prompt.
 *
 * Why: on slow devices a thinking model can burn its whole token budget in
 * <think> — this is the escape hatch, offered once the block passes 8 lines.
 * The aborted assistant turn is never pushed to history (only completed
 * replies are), so the retry sees a clean transcript.
 */
async function justAnswerMe(event) {
    event.stopPropagation();

    // Abort current generation and tell server to stop
    if (abortController) {
        abortController.abort();
    }
    fetch('/v1/cancel', { method: 'POST' }).catch(() => {});

    // Find the last user message
    let lastUserMsg = null;
    for (let i = chatHistory.length - 1; i >= 0; i--) {
        if (chatHistory[i].role === 'user') {
            lastUserMsg = chatHistory[i];
            break;
        }
    }
    if (!lastUserMsg) return;

    // Remove the aborted assistant message from history (it was never complete)
    // The DOM bubble will stay but we'll add a new one below

    // Wait for abort to settle
    await new Promise(r => setTimeout(r, 100));

    // Mark the old bubble as cancelled
    const lastBubble = chat.querySelector('.message.assistant:last-child');
    if (lastBubble) {
        const meta = lastBubble.querySelector('.meta');
        if (!meta) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'meta';
            metaDiv.innerHTML = '<span style="color:var(--text-dim)">[retrying without thinking]</span>';
            lastBubble.appendChild(metaDiv);
        }
    }

    // Create new assistant bubble and send
    const assistantDiv = addMessage('assistant', '');
    assistantDiv.innerHTML = '<span class="typing-indicator"></span>';
    setGenerating(true);
    const t0 = performance.now();

    try {
        abortController = new AbortController();
        const resp = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: abortController.signal,
            body: JSON.stringify(buildRequestBody({
                messages: [
                    { role: 'system', content: NO_THINK_PROMPT },
                    ...chatHistory.filter(m => m.role !== 'system'),
                ],
            })),
        });

        const respDevice = resp.headers.get('X-Device') || '';
        device = respDevice;

        if (!resp.ok) {
            const err = await resp.json();
            assistantDiv.innerHTML = `<span style="color:var(--error)">${escapeHtml(err.error?.message || 'Error')}</span>`;
            return;
        }

        const contentType = resp.headers.get('content-type') || '';
        if (contentType.includes('text/event-stream')) {
            let fullText = '';
            const think = { open: false };  // reasoning_content -> <think> re-tagging state
            resetStreamState();
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;
                    try {
                        const chunk = JSON.parse(data);
                        const piece = appendDelta(think, chunk.choices?.[0]?.delta || {});
                        if (piece) {
                            fullText += piece;
                            updateStreamBubble(assistantDiv, fullText);
                        }
                    } catch {}
                }
            }

            fullText += closeThink(think);
            assistantDiv.innerHTML = renderMarkdown(fullText, false);
            resetStreamState();
            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
            const metaDiv = document.createElement('div');
            metaDiv.className = 'meta';
            metaDiv.innerHTML = respDevice
                ? `<span class="device-tag">${respDevice}</span> ${elapsed}s (no-think)`
                : `${elapsed}s (no-think)`;
            assistantDiv.appendChild(metaDiv);
            chatHistory.push({ role: 'assistant', content: fullText });
        } else {
            const data = await resp.json();
            const text = data.choices?.[0]?.message?.content || '';
            assistantDiv.innerHTML = renderMarkdown(text, false);
            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
            const metaDiv = document.createElement('div');
            metaDiv.className = 'meta';
            metaDiv.innerHTML = respDevice
                ? `<span class="device-tag">${respDevice}</span> ${elapsed}s (no-think)`
                : `${elapsed}s (no-think)`;
            assistantDiv.appendChild(metaDiv);
            chatHistory.push({ role: 'assistant', content: text });
        }
    } catch (err) {
        if (err.name !== 'AbortError') {
            assistantDiv.innerHTML = `<span style="color:var(--error)">${escapeHtml(err.message)}</span>`;
        }
    } finally {
        setGenerating(false);
        abortController = null;
        input.focus();
        updateHistoryWarning();
    }
}
window.justAnswerMe = justAnswerMe;

// --- Chat ---

/**
 * Append a chat bubble. String content goes through renderMarkdown; anything
 * else is trusted, pre-built HTML (callers escape it themselves — see
 * sendMessage's attached-image path for why that trust exists).
 */
function addMessage(role, content, meta) {
    const div = document.createElement('div');
    div.className = `message ${role}`;

    if (typeof content === 'string') {
        div.innerHTML = renderMarkdown(content);
    } else {
        div.innerHTML = content;
    }

    if (meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'meta';
        metaDiv.innerHTML = meta;
        div.appendChild(metaDiv);
    }

    chat.appendChild(div);
    scrollToBottom();
    return div;
}

/**
 * Model text -> bubble HTML: split out the <think> block, render both halves
 * as markdown.
 *
 * The four think states below (paired tags / orphan closer / still-open /
 * tag-chars-arriving) exist because chat templates differ in WHERE the
 * opening tag lives — some pre-seed it into the prompt so only the closer is
 * ever generated. Each state's comment carries its evidence.
 */
function renderMarkdown(text, isStreaming) {
    // Handle <think>...</think> blocks BEFORE escaping HTML
    // These are raw model output tags, not user HTML
    let thinkHtml = '';
    let mainText = text;

    // Complete: <think>...</think> followed by the actual answer
    let thinkMatch = text.match(/^<think>([\s\S]*?)<\/think>\s*([\s\S]*)$/);
    // Orphan closing tag: a CLOSING </think> with no opening one. This is not
    // a model quirk — the chat template pre-seeds the opening tag into the
    // PROMPT as the assistant's generation prefix, so it never appears in the
    // generated text. Qwen3.8's template ends with:
    //     {{- '<|im_start|>assistant\n' }}  ... {{- '<think>\n' }}
    // The model therefore starts generating already inside the block and only
    // ever emits the closer. Both Qwen3.8 and SmolLM3 reason fully here; the
    // reasoning is real, it was just being rendered as the answer along with a
    // literal "</think>". Observed on the B60, 2026-08-15.
    let thinkClose = !thinkMatch && text.match(/^([\s\S]*?)<\/think>\s*([\s\S]*)$/);
    // Partial: <think> started but no closing tag yet (streaming)
    let thinkOpen = !thinkMatch && !thinkClose && text.match(/^<think>([\s\S]*)$/);
    // Very early: just the opening tag arriving character by character
    let thinkStarting = !thinkMatch && !thinkClose && !thinkOpen && /^<(?:t(?:h(?:i(?:n(?:k)?)?)?)?)?$/.test(text.trim());

    if (thinkMatch) {
        const thinkContent = thinkMatch[1].trim();
        mainText = thinkMatch[2].trim();
        // Skip empty think blocks (no-think mode sometimes emits empty tags)
        if (thinkContent) {
            thinkHtml = renderThinkingBlock(thinkContent, false, thinkExpanded ? '' : 'collapsed');
        }
    } else if (thinkClose) {
        // Same handling as the paired case. Note this only settles once the
        // closer arrives: mid-stream we cannot tell a pre-seeded thinker from
        // a model that simply never thinks, so the text streams as the answer
        // and snaps into a collapsed block at </think>. That is the safe way
        // round — the alternative hides a non-thinking model's whole reply.
        const thinkContent = thinkClose[1].trim();
        mainText = thinkClose[2].trim();
        if (thinkContent) {
            thinkHtml = renderThinkingBlock(thinkContent, false, thinkExpanded ? '' : 'collapsed');
        }
    } else if (thinkOpen) {
        // Still thinking — show content live
        const thinkContent = thinkOpen[1].trim();
        if (thinkContent) {
            const lines = thinkContent.split('\n');
            if (lines.length > 4) {
                // Enough lines — expandable + just-answer button
                const justAnswerBtn = lines.length > 8
                    ? `<button class="just-answer" data-just-answer>Just answer me, dammit!</button>`
                    : '';
                const cls = thinkExpanded ? '' : 'collapsed';
                thinkHtml = renderThinkingBlock(thinkContent, true, cls) + justAnswerBtn;
            } else {
                // Few lines — show all, no collapse needed
                thinkHtml = `<div class="think-block streaming">
                    <div class="think-header">Thinking...</div>
                    <div class="think-preview">${mdEscapeAndRender(thinkContent)}</div>
                </div>`;
            }
        } else {
            thinkHtml = `<div class="think-block streaming">
                <div class="think-header">Thinking...</div>
            </div>`;
        }
        mainText = '';
    } else if (thinkStarting && isStreaming) {
        // Partial <think> tag still arriving
        thinkHtml = `<div class="think-block streaming">
            <div class="think-header">Thinking...</div>
        </div>`;
        mainText = '';
    }

    // Render the main text as markdown
    return thinkHtml + mdEscapeAndRender(mainText);
}

/**
 * Renders the inner HTML for a thinking block's full/preview content.
 * Uses the same markdown renderer as the main answer so markdown syntax
 * inside thinking (headers, lists, bold, code blocks) is rendered, not
 * leaked as raw text. Preview = the last 3 lines (the live tail).
 */
function renderThinkingBlock(content, streaming, extraClass) {
    const full = mdEscapeAndRender(content);
    const preview = mdEscapeAndRender(content.split('\n').slice(-3).join('\n'));
    const cls = streaming ? 'streaming ' + extraClass : extraClass;
    return `<div class="think-block ${cls}" data-think-toggle>
        <div class="think-header">Thinking... <span class="think-toggle">(click to expand)</span></div>
        <div class="think-full">${full}</div>
        <div class="think-preview">${preview}</div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Markdown rendering (self-contained, no dependencies)
// ---------------------------------------------------------------------------
/**
 * Escaped-markdown renderer (hand-rolled, no library — PR #23).
 *
 * Order-of-operations:
 *   1. escapeHtml the whole input to guarantee raw model HTML can never execute.
 *      All passes below operate on the escaped text.
 *   2. Pull fenced code blocks into protected placeholders so the inline
 *      bold/italic/code passes below never touch code content (the old
 *      renderer mangled `**x**` and `*x*` inside code blocks).
  *   3. Block passes, line by line: code blocks, headers, blockquotes, lists,
  *      tables, paragraphs. Each line's text goes through mdInline exactly ONCE, inside
 *      the block pass — running it on the whole text first and again per line
 *      corrupts _ and * inside href/src attributes generated by the first
 *      pass, and lets *...* match across newlines, eating `* ` bullet lists.
 *   4. Inline passes (mdInline): inline code, images, links (scheme-checked),
 *      bold, italic — with code spans and attribute values protected.
 */
function mdEscapeAndRender(text) {
    if (!text) return '';
    let html = escapeHtml(text);
    const codeBlocks = [];
    html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
        const i = codeBlocks.length;
        codeBlocks.push({ lang, code: code.replace(/\n$/, '') });
        return `\u0000CODE${i}\u0000`;
    });
    const lines = html.split('\n');
    const out = [];
    let inList = false, inBlockquote = false, inParagraph = false;
    /** Close the open <p>/<ol|ul>/<blockquote> if any — each block pass calls
        the ones its element cannot nest inside. */
    const closePara = () => { if (inParagraph) { out.push('</p>'); inParagraph = false; } };
    /** See closePara. */
    const closeList = () => { if (inList) { out.push(`</${inList}>`); inList = false; } };
    /** See closePara. */
    const closeBlockquote = () => { if (inBlockquote) { out.push('</blockquote>'); inBlockquote = false; } };
    for (let i = 0; i < lines.length; i++) {
        const trimmed = lines[i].trim();
        const codeMatch = trimmed.match(/^(\u0000CODE\d+\u0000)$/);
        // Guard: literal NUL+CODEn+NUL in model text has no matching block --
        // fall through to the paragraph path instead of crashing on blk.lang.
        const codeBlk = codeMatch && codeBlocks[+codeMatch[1].match(/\d+/)];
        if (codeBlk) {
            closePara(); closeList(); closeBlockquote();
            out.push(`<pre><code class="language-${escapeHtml(codeBlk.lang || '')}">${codeBlk.code}</code><button class="copy-btn" onclick="copyCode(this)">copy</button></pre>`);
            continue;
        }
        const h = trimmed.match(/^(#{1,6})\s+(.*)$/);
        if (h) { closePara(); closeList(); closeBlockquote(); out.push(`<h${h[1].length}>${mdInline(h[2])}</h${h[1].length}>`); continue; }
        // Input is already HTML-escaped, so markdown "> quote" arrives here
        // as "&gt; quote" -- match the entity, not the raw >.
        const bq = trimmed.match(/^&gt; ?(.*)$/);
        if (bq) { closePara(); closeList(); if (!inBlockquote) { closeBlockquote(); out.push('<blockquote>'); inBlockquote = true; } out.push(mdInline(bq[1])); continue; }
        closeBlockquote();
        // Tables: `| a | b |` immediately followed by a |-|-| separator row.
        // The whole table is consumed in one go (we already hold every line);
        // each cell runs through mdInline exactly once.
        if (i + 1 < lines.length && trimmed.startsWith('|') && trimmed.endsWith('|') && isTableSeparator(lines[i + 1].trim())) {
            closePara(); closeList();
            out.push('<table><thead><tr>' + splitTableRow(trimmed).map(c => `<th>${mdInline(c)}</th>`).join('') + '</tr></thead><tbody>');
            let j = i + 2;
            while (j < lines.length) {
                const cellLine = lines[j].trim();
                if (!(cellLine.startsWith('|') && cellLine.endsWith('|'))) break;
                out.push('<tr>' + splitTableRow(cellLine).map(c => `<td>${mdInline(c)}</td>`).join('') + '</tr>');
                j++;
            }
            out.push('</tbody></table>');
            i = j - 1;
            continue;
        }
        const ol = trimmed.match(/^(\d+)\. (.*)$/);
        const ul = trimmed.match(/^[-*+] (.*)$/);
        if (ol || ul) {
            closePara();
            const listType = ol ? 'ol' : 'ul';
            if (inList !== listType) { closeList(); out.push(`<${listType}>`); inList = listType; }
            out.push(`<li>${mdInline((ol ? ol[2] : ul[1]) || '')}</li>`);
            continue;
        }
        closeList();
        if (trimmed === '') { closePara(); closeBlockquote(); continue; }
        if (!inParagraph) { out.push('<p>'); inParagraph = true; } else { out.push(' '); }
        out.push(mdInline(lines[i]));
    }
    closePara(); closeList(); closeBlockquote();
    return out.join('').replace(/\u0000CODE\d+\u0000/g, m => {
        const blk = codeBlocks[+(m.match(/\d+/) || [0])[0]];
        if (!blk) return ''; // literal NUL junk in model text, not ours
        return `<pre><code class="language-${escapeHtml(blk.lang || '')}">${blk.code}</code><button class="copy-btn" onclick="copyCode(this)">copy</button></pre>`;
    });
}

/**
 * Apply inline passes (on already-escaped text) to a single LINE of text.
 * Called exactly once per line by the block pass in mdEscapeAndRender — never
 * run it twice on the same text: a second pass corrupts _ and * inside the
 * href/src attributes the first pass generated.
 * Code spans and generated tags are pulled into placeholders so the emphasis
 * passes can't touch code content (`snake_case`) or URLs (...model_id...).
 */
function mdInline(str) {
    const guarded = [];
    /** Park generated HTML behind a NUL placeholder the emphasis regexes can't touch. */
    const guard = (html) => { guarded.push(html); return `\u0000G${guarded.length - 1}\u0000`; };
    let s = str;
    s = s.replace(/`([^`\n]+)`/g, (_, c) => guard(`<code>${c}</code>`));
    s = s.replace(/!\[([^\]\n]*)\]\(([^)\n]+)\)/g, (m, a, u) => {
        const url = safeUrl(u);
        return url ? guard(`<img alt="${escapeAttr(a)}" src="${url}">`) : m;
    });
    s = s.replace(/\[([^\]\n]+)\]\(([^)\n]+)\)/g, (m, l, u) => {
        const url = safeUrl(u);
        // Only the opening tag is guarded — emphasis inside link text still works.
        return url ? guard(`<a href="${url}">`) + l + '</a>' : m;
    });
    s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');
    s = s.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    s = s.replace(/_([^_\n]+)_/g, '<em>$1</em>');
    return s.replace(/\u0000G(\d+)\u0000/g, (m, i) => guarded[+i] !== undefined ? guarded[+i] : '');
}

/**
 * escapeHtml (textContent -> innerHTML) escapes & < > but NOT quotes. That is
 * fine for text nodes, but attribute values built from model output must have
 * quotes escaped too, or `![x" onerror=...](y)` breaks out of the attribute.
 */
function escapeAttr(s) {
    return s.replace(/"/g, '&quot;');
}

/**
 * Allow only URL schemes that cannot execute script (plus scheme-less
 * relative/fragment URLs). javascript:, data:, vbscript: etc. are rejected;
 * the caller then leaves the markdown as plain, already-escaped text.
 * Why escapeHtml is not enough: every character in a javascript: URI is
 * escape-neutral, so the value survives escaping intact and executes on click.
 */
function safeUrl(u) {
    const url = u.trim();
    const scheme = url.match(/^[a-zA-Z][a-zA-Z0-9+.-]*:/);
    if (scheme && !/^(https?|mailto):$/i.test(scheme[0])) return null;
    return escapeAttr(url);
}


/**
 * HTML-escape via the DOM (textContent -> innerHTML): & < > only — quote
 * escaping for attributes is escapeAttr's job.
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
// --- tables ---

/**
 * Split a `| a | b |` row into trimmed cells.
 *
 * Why: a bare `row.split('|')` breaks on `| select a | b from t |` — a pipe
 * inside a backtick code span is a literal cell character, not a column
 * delimiter — and on `| [x](http://h.com/p|q) |` where the pipe is inside a
 * link URL. We walk the row and only split on `|` when outside a `...` code
 * span AND outside a `[...](...)` link/image group.
 *
  * In: a string optionally framed by `|`. Out: array of trimmed cell strings.
  * A `[` with no matching `]` does not pin the parser in link mode: the next
  * `|` resets bracket depth and still acts as a cell separator, so the rest of
  * the row is not swallowed into one cell.
 */
function splitTableRow(row) {
    let s = row.trim();
    if (s.startsWith('|')) s = s.slice(1);
    if (s.endsWith('|')) s = s.slice(0, -1);
    const cells = [];
    let cur = '';
    let inCode = false;
    // Track bracket depth so an unclosed `[` (e.g. a stray `[` with no matching
    // `]`) does not pin inLink=true and swallow every subsequent `|` as link
    // URL text. A `|` inside `[...]` is preserved; a `|` after an unbalanced
    // `[` still acts as a cell separator. mdInline uses bounded regexes that
    // can only ever match a complete `[text](url)`, so it is already safe.
    let bracketDepth = 0;
    let inParens = false;  // inside the (url) of a `[...](url)` link
    for (let i = 0; i < s.length; i++) {
        const ch = s[i];
        if (ch === '`') { inCode = !inCode; cur += ch; continue; }
        if (inCode) { cur += ch; continue; }
        if (ch === '[') { bracketDepth++; cur += ch; continue; }
        if (ch === ']' && bracketDepth > 0) {
            bracketDepth = Math.max(0, bracketDepth - 1);
            // `]` closes the link text; only a following `(` opens the URL
            // destination. Footnote-style `[1]` has no `(`, so it must not pin
            // inParens — otherwise the next `|` is swallowed as URL text.
            if (s[i + 1] === '(') inParens = true;
            cur += ch;
            continue;
        }
        if (ch === ')' && inParens) { inParens = false; cur += ch; continue; }
        if (ch === '|') {
            // A `|` always closes any open `[` never matched by `]`, so a
            // stray/unclosed bracket does not eat the rest of the row. The URL
            // part of a `[text](url)` link is tracked by inParens, so a pipe
            // inside the URL is preserved as link text.
            if (bracketDepth > 0) inParens = false;
            bracketDepth = 0;
            if (inParens) { cur += ch; continue; }
            cells.push(cur.trim()); cur = '';
            continue;
        }
        cur += ch;
    }
    cells.push(cur.trim());
    return cells;
}

// A separator row (| --- |, | :---: |, |---|---| ...). Only dashes and
// optional alignment colons per cell — after escaping, - : | survive intact.
/**
 * True iff `line` is a Markdown table separator row (`|---|`, `|:--:|`, ...).
 *
 * Why: the table block in mdEscapeAndRender must distinguish a header/separator
 * pair from an ordinary `| a | b |` paragraph. Splitting on `|` and requiring
 * every cell to match `^\s*:?-+:?\s*$` rejects text cells and accepts the
 * alignment-colon variants.
 *
 * In: a trimmed-or-not string. Out: boolean — false for '', non-framed rows,
 * or any cell that is not pure dashes/colons.
 */
function isTableSeparator(line) {
    const s = line.trim();
    if (s === '' || !s.startsWith('|') || !s.endsWith('|')) return false;
    return s.slice(1, -1).split('|').every(c => /^\s*:?-+:?\s*$/.test(c));
}


/** Copy a code block's text; button label doubles as the confirmation. */
function copyCode(btn) {
    const code = btn.parentElement.querySelector('code').textContent;
    navigator.clipboard.writeText(code);
    btn.textContent = 'copied';
    setTimeout(() => btn.textContent = 'copy', 1500);
}
// Make copyCode available globally
window.copyCode = copyCode;
// Exposed for browser-test harnesses (no functional side effects)
window.mdEscapeAndRender = mdEscapeAndRender;
window.renderMarkdown = renderMarkdown;
window.updateStreamBubble = updateStreamBubble;
window.resetStreamState = resetStreamState;

/**
 * Send the composed turn and stream the reply into a new assistant bubble.
 *
 * Contract with history: the user turn is pushed BEFORE the request but
 * removed again (dropUserMsg) on any failure the model never saw — a stale
 * user turn silently prepended to every later request reads as a corrupted
 * transcript (observed on Glimmer: it answers the wrong turn). A deliberate
 * cancel keeps it: justAnswerMe's retry needs it there.
 */
async function sendMessage() {
    const text = input.value.trim();
    if (!text && !attachedImage) return;
    if (isGenerating) return;
    thinkExpanded = false; // reset for new message

    // Build user message content
    let userContent;
    let displayHtml = '';

    if (attachedImage) {
        userContent = [];
        if (text) userContent.push({ type: 'text', text: text });
        userContent.push({ type: 'image_url', image_url: { url: attachedImage } });
        displayHtml = escapeHtml(text);
        displayHtml += `<img src="${attachedImage}" alt="attached">`;
    } else {
        userContent = text;
        displayHtml = escapeHtml(text);
    }

    // Show user message. Bypass renderMarkdown: displayHtml is already
    // escaped where needed, and with an attached image it contains a real
    // <img> tag — markdown rendering would escape it and dump the base64
    // src as visible text.
    const userDiv = addMessage('user', '');
    userDiv.innerHTML = displayHtml.replace(/\n/g, '<br>');
    const userMsg = { role: 'user', content: userContent };
    chatHistory.push(userMsg);
    updateHistoryWarning();
    /** A send the model never saw (server not ready, network error) must not
     * stay in history: it becomes a stale user turn silently prepended to
     * every later request — two consecutive user messages read as a corrupted
     * transcript to the model (observed on Glimmer: it answers the wrong turn). */
    const dropUserMsg = () => {
        const i = chatHistory.lastIndexOf(userMsg);
        if (i >= 0) chatHistory.splice(i, 1);
    };

    // Clear input
    input.value = '';
    input.style.height = 'auto';
    clearImage();

    // Create assistant bubble with waiting indicator
    const assistantDiv = addMessage('assistant', '');
    assistantDiv.innerHTML = '<span class="typing-indicator"></span>';
    setGenerating(true);
    const t0 = performance.now();

    // After 3s with no response, check if a model is reloading and show that
    const reloadCheckTimer = setTimeout(async () => {
        try {
            const r = await fetch('/health');
            const data = await r.json();
            const reloading = Object.values(data.devices || {}).some(
                d => d.status === 'loading' || d.status === 'warming_up'
            );
            if (reloading && isGenerating) {
                assistantDiv.innerHTML =
                    '<span class="typing-indicator"></span> ' +
                    '<span style="color:var(--text-dim)">Reloading model…</span>';
            }
        } catch {}
    }, 3000);

    try {
        abortController = new AbortController();
        const resp = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: abortController.signal,
            body: JSON.stringify(buildRequestBody()),
        });
        clearTimeout(reloadCheckTimer);

        const model = resp.headers.get('X-Model') || '';
        device = resp.headers.get('X-Device') || '';

        if (!resp.ok) {
            dropUserMsg();
            const err = await resp.json();
            assistantDiv.innerHTML = `<span style="color:var(--error)">${escapeHtml(err.error?.message || 'Error')}</span>`;
            return;
        }

        // Check if streaming (SSE) or single response
        const contentType = resp.headers.get('content-type') || '';
        if (contentType.includes('text/event-stream')) {
            // Streaming
            let fullText = '';
            const think = { open: false };  // reasoning_content -> <think> re-tagging state
            resetStreamState();
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep incomplete line

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;
                    try {
                        const chunk = JSON.parse(data);
                        const piece = appendDelta(think, chunk.choices?.[0]?.delta || {});
                        if (piece) {
                            fullText += piece;
                            updateStreamBubble(assistantDiv, fullText);
                        }
                    } catch {}
                }
            }

            // Re-render with streaming=false to collapse think block
            fullText += closeThink(think);
            assistantDiv.innerHTML = renderMarkdown(fullText, false);
            resetStreamState();

            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
            const metaHtml = device
                ? `<span class="device-tag">${device}</span> ${elapsed}s`
                : `${elapsed}s`;
            const metaDiv = document.createElement('div');
            metaDiv.className = 'meta';
            metaDiv.innerHTML = metaHtml;
            assistantDiv.appendChild(metaDiv);

            chatHistory.push({ role: 'assistant', content: fullText });
        } else {
            // Non-streaming (VLM)
            const data = await resp.json();
            const text = data.choices?.[0]?.message?.content || '';
            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

            assistantDiv.innerHTML = renderMarkdown(text);
            const metaHtml = device
                ? `<span class="device-tag">${device}</span> ${elapsed}s`
                : `${elapsed}s`;
            const metaDiv = document.createElement('div');
            metaDiv.className = 'meta';
            metaDiv.innerHTML = metaHtml;
            assistantDiv.appendChild(metaDiv);

            chatHistory.push({ role: 'assistant', content: text });
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            // Deliberate cancel: keep the user message — justAnswerMe's retry
            // (and a manual re-ask) still needs it in history.
            assistantDiv.innerHTML += '<br><span style="color:var(--text-dim)">[cancelled]</span>';
        } else {
            dropUserMsg();
            assistantDiv.innerHTML = `<span style="color:var(--error)">${escapeHtml(err.message)}</span>`;
        }
    } finally {
        setGenerating(false);
        abortController = null;
        input.focus();
        updateHistoryWarning();
    }
}

// --- History-length warning ---
// NPU slots cap the prompt at MAX_PROMPT_LEN=4096 tokens (~16k chars). There's
// no client-side tokenizer, so this nudges on a rough ~4 chars/token estimate
// when accumulated history passes ~12000 chars (~3k tokens, near the cap).
// Approximate by design — it's a warning, not an enforcement.
// The banner is NPU-only: only NPU has a hard prompt-token cap, so init only
// proceeds when /health reports an NPU slot. Styles live in style.css
// (CSS variables --warn-*) — no JS <style> injection.
const HISTORY_WARN_CHARS = 12000;

/**
 * Sum the characters of every message currently in chatHistory.
 *
 * Why: the NPU prompt cap is a character/token ceiling, and the only client-side
 * signal we have is history size — there is no tokenizer in the browser.
 *
 * In: nothing (reads the module-level `chatHistory`). Out: integer char count;
 * string content is counted directly, and multimodal `Array` content counts each
 * part's `.text` (images contribute nothing to the token estimate).
 */
function historyCharCount() {
    let chars = 0;
    for (const m of chatHistory) {
        const c = m.content;
        if (typeof c === 'string') {
            chars += c.length;
        } else if (Array.isArray(c)) {
            for (const part of c) {
                if (part && typeof part.text === 'string') chars += part.text.length;
            }
        }
    }
    return chars;
}

/**
 * Show or hide the history-length banner based on historyCharCount().
 *
 * Why: a long session silently nears the NPU prompt cap; the banner is the only
 * user-visible nudge to start a fresh chat before truncation bites. No-op when
 * the banner does not exist (e.g. /health reported no NPU device).
 *
 * In: nothing. Out: nothing; toggles `hidden` on #history-warning if present.
 */
function updateHistoryWarning() {
    const warn = document.getElementById('history-warning');
    if (warn) warn.hidden = historyCharCount() < HISTORY_WARN_CHARS;
}

/**
 * Build the history-length banner and insert it above the chat input.
 * Idempotent.
 *
 * Why: the banner is NPU-only — only NPU has a hard prompt-token cap that
 * truncation can silently break. On non-NPU devices the warning is noise, so
 * init gates on /health reporting an NPU slot. Styles are in style.css via
 * --warn-* CSS variables, so no <style> is injected from JS.
 *
 * In: nothing. Out: nothing; appends a `.history-warning` element (hidden) to
 * `.input-area .input-wrap`. Returns early if /health has no NPU device, the
 * host is missing, or the banner already exists.
 */
async function initHistoryWarning() {
    // Gate on NPU availability: only NPU has a prompt-token cap.
    try {
        const r = await fetch('/health');
        const data = await r.json();
        if (!data.devices || !data.devices.npu) return;
    } catch {
        // /health failed — don't block the UI; skip the banner.
        return;
    }

    const host = document.querySelector('.input-area .input-wrap');
    if (!host || document.getElementById('history-warning')) return;

    const warn = document.createElement('div');
    warn.className = 'history-warning';
    warn.id = 'history-warning';
    warn.hidden = true;

    const span = document.createElement('span');
    span.textContent = 'Long chat — the NPU model may start truncating.';

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = 'Start a new chat (Ctrl+N)';
    btn.addEventListener('click', newChat);

    warn.appendChild(span);
    warn.appendChild(btn);
    host.insertBefore(warn, host.firstChild);
}

/**
 * Clear history + chat pane (Ctrl+N). Also the user's tool for keeping long
 * sessions under the NPU's MAX_PROMPT_LEN — history is unbounded by design.
 */
function newChat() {
    chatHistory = [];
    chat.innerHTML = '';
    input.focus();
    updateHistoryWarning();
}

// --- Image handling ---

/**
 * Stage an image (file picker, paste, or drop) as a base64 data URI for the
 * next send; the focus hand-off below is the why of the last line.
 */
function attachImage(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = () => {
        attachedImage = reader.result;
        previewImg.src = attachedImage;
        imagePreview.style.display = 'block';
        // Attach-image-first leaves focus on the attach button (or wherever
        // the paste happened) — keystrokes then go nowhere and the input
        // feels dead. Hand focus to the box the user will type in next.
        input.focus();
    };
    reader.readAsDataURL(file);
}

/** Drop the staged image and hide its preview. */
function clearImage() {
    attachedImage = null;
    previewImg.src = '';
    imagePreview.style.display = 'none';
}

// --- Event listeners ---

// Event delegation for think blocks (survives DOM re-renders)
chat.addEventListener('click', (e) => {
    // "Just answer me" button
    if (e.target.closest('[data-just-answer]')) {
        e.stopPropagation();
        justAnswerMe(e);
        return;
    }
    // Think block expand/collapse
    const thinkBlock = e.target.closest('[data-think-toggle]');
    if (thinkBlock) {
        thinkExpanded = !thinkExpanded;
        thinkBlock.classList.toggle('collapsed');
    }
});

// Send
sendBtn.addEventListener('click', () => {
    if (isGenerating) cancelGeneration();
    else sendMessage();
});

// No-think defaults ON (slow devices + thinking models = runaway loops);
// the user's choice sticks across sessions.
noThinkCheckbox.checked = localStorage.getItem('nollama-no-think') !== 'off';
noThinkCheckbox.addEventListener('change', () => {
    localStorage.setItem('nollama-no-think', noThinkCheckbox.checked ? 'on' : 'off');
});
input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 150) + 'px';
});

// New chat
newChatBtn.addEventListener('click', newChat);

// File attach
attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) attachImage(fileInput.files[0]);
    fileInput.value = '';
});
removeImageBtn.addEventListener('click', clearImage);

// Paste image
document.addEventListener('paste', (e) => {
    for (const item of e.clipboardData.items) {
        if (item.type.startsWith('image/')) {
            e.preventDefault();
            attachImage(item.getAsFile());
            return;
        }
    }
    // Images copied from web pages often arrive as a text/html flavor with a
    // data: URL and no image/ item — without this, the whole base64 string
    // lands in the input box as text.
    const dataUrl = (e.clipboardData.getData('text/html') + ' ' +
                     e.clipboardData.getData('text/plain'))
        .match(/data:image\/(?:png|jpe?g|gif|webp);base64,[A-Za-z0-9+/=]+/);
    if (dataUrl) {
        e.preventDefault();
        attachedImage = dataUrl[0];
        previewImg.src = attachedImage;
        imagePreview.style.display = 'block';
        input.focus();
    }
});

// Drag and drop
let dragCounter = 0;
document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    if (e.dataTransfer.types.includes('Files')) {
        dropOverlay.classList.add('active');
    }
});
document.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) {
        dropOverlay.classList.remove('active');
        dragCounter = 0;
    }
});
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => {
    e.preventDefault();
    dropOverlay.classList.remove('active');
    dragCounter = 0;
    const file = e.dataTransfer.files[0];
    if (file) attachImage(file);
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'n') {
        e.preventDefault();
        newChat();
    }
    if (e.key === 'Escape' && isGenerating && abortController) {
        abortController.abort();
        fetch('/v1/cancel', { method: 'POST' }).catch(() => {});
    }
});

// --- Start ---
initHistoryWarning();
init();
