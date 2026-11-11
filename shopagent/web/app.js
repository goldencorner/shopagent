(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const ICON = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
         stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M4 7h16l-1.4 12.1a2 2 0 0 1-2 1.9H7.4a2 2 0 0 1-2-1.9L4 7Z"/>
      <path d="M9 7V6a3 3 0 0 1 6 0v1"/>
      <circle cx="16.6" cy="12.4" r="2.4"/>
      <path d="m18.6 14.4 1.9 1.9"/>
    </svg>`;

  const EXAMPLES = [
    { label: "推荐几款瑜伽垫", query: "推荐几款瑜伽垫" },
    { label: "两人露营帐篷，$100 以内", query: "推荐一款适合两个人用的露营帐篷，预算 $100 以内" },
    { label: "对比两款露营帐篷", query: "对比一下 CAMEL CROWN 和 Forceatt 的露营帐篷，哪个更划算" },
    { label: "有没有卖猫粮的", query: "有没有卖猫粮的" },
  ];

  const INTENT_LABEL = { shop: "购物", compare: "对比", chitchat: "闲聊" };

  const messagesEl = $("#messages");
  const inner = document.createElement("div");
  inner.className = "messages-inner";
  messagesEl.appendChild(inner);

  const inputEl = $("#input");
  const composerEl = $("#composer");
  const sendBtn = $("#send");
  const statusEl = $("#status");
  const statusText = $("#status-text");
  const configPanel = $("#config-panel");
  const configMode = $("#config-mode");
  const configNoteEl = $("#config-note");
  const configNoteText = $("#config-note-text");
  const configRange = $("#config-range");
  const configEditValue = $("#config-edit-value");
  const funnelRecall = $("#funnel-recall");
  const funnelRerank = $("#funnel-rerank");
  const funnelRerankValue = $("#funnel-rerank-value");
  const funnelOutput = $("#funnel-output");
  const sessionsEl = $("#sessions");
  const sessionCount = $("#session-count");
  const profileEl = $("#profile");
  const examplesEl = $("#examples");
  const traceToggle = $("#show-trace");
  const modelSelect = $("#model-select");
  const modelStatusEl = $("#model-status");
  const modelStatusText = $("#model-status-text");
  const modelRefreshBtn = $("#model-refresh");
  const stopBtn = $("#stop");

  const mainEl = $("#main");
  const viewTitle = $("#view-title");
  const subtitle = $("#subtitle");
  const tabChat = $("#tab-chat");
  const tabCatalog = $("#tab-catalog");
  const catalogEl = $("#catalog");
  const catalogStatsEl = $("#catalog-stats");
  const priceBarsEl = $("#price-bars");
  const brandBarsEl = $("#brand-bars");
  const priceNote = $("#price-note");
  const brandNote = $("#brand-note");
  const catalogQ = $("#catalog-q");
  const catalogPrice = $("#catalog-price");
  const catalogSort = $("#catalog-sort");
  const catalogCount = $("#catalog-count");
  const catalogSource = $("#catalog-source");
  const productGrid = $("#product-grid");
  const catalogPager = $("#catalog-pager");
  const productModal = $("#product-modal");
  const modalTitle = $("#modal-title");
  const modalKicker = $("#modal-kicker");
  const modalBody = $("#modal-body");
  const modalClose = $("#modal-close");

  const state = {
    sessionId: readStorage("shopagent.session") || newSessionId(),
    log: [],
    sending: false,
    showTrace: readStorage("shopagent.trace") === "1",
    model: readStorage("shopagent.model") || "deepseek",
  };
  writeStorage("shopagent.session", state.sessionId);

  // 停止生成：当前请求的 AbortController + 是否由用户主动停止
  let activeAbort = null;
  let stopRequested = false;

  const catalogState = {
    statsLoaded: false,
    loaded: false,
    requestId: 0,
    page: 1,
    query: "",
    price: "all",
    sort: "default",
  };
  let detailRequestId = 0;
  const nf = new Intl.NumberFormat("zh-CN");
  const moneyFmt = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });

  // ---------- storage / ids ----------

  function readStorage(key) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  function writeStorage(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* 隐私模式或存储被禁用时不影响使用 */
    }
  }

  function newSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  }

  // ---------- tiny render helpers ----------

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (ch) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[ch]
    ));
  }

  function titleCandidates(title) {
    const full = String(title || "").trim();
    if (!full) return [];

    const out = new Set([full]);
    const comma = full.indexOf(",");
    if (comma >= 10) out.add(full.slice(0, comma).trim());

    // 模型经常在词边界把长标题截断（如 "…Thin Hot Yoga Mat" 之后就省略），
    // 所以每个词边界之前的部分都作为候选；长的优先，尽量匹配到最多信息。
    let prefix = "";
    for (const word of full.split(/\s+/)) {
      prefix = prefix ? `${prefix} ${word}` : word;
      out.add(prefix);
    }
    if (full.length > 70) out.add(full.slice(0, 70).trim());

    return [...out]
      .map((item) => item.trim())
      .filter((item) => item.length >= 20 && item.length <= full.length)
      .sort((a, b) => b.length - a.length);
  }

  function formatAnswer(text, products) {
    let html = escapeHtml(text || "");
    const slots = [];
    const linked = new Set();

    if (Array.isArray(products) && products.length) {
      const refs = products
        .filter((item) => item && item.asin && item.title)
        .map((item) => ({ ...item, escaped: escapeHtml(item.title) }))
        .sort((a, b) => b.escaped.length - a.escaped.length);

      // 标题可能在正文里出现多次；只链接唯一匹配的候选，避免把 A 的标题
      // 链到 B。短标题（逗号前的品牌+型号）作为模型省略全称时的兜底。
      const counts = new Map();
      refs.forEach((ref) => {
        titleCandidates(ref.title).forEach((candidate) => {
          const key = escapeHtml(candidate);
          counts.set(key, (counts.get(key) || 0) + 1);
        });
      });

      refs.forEach((ref) => {
        for (const candidate of titleCandidates(ref.title)) {
          const escaped = escapeHtml(candidate);
          if (counts.get(escaped) !== 1) continue;
          const index = html.indexOf(escaped);
          if (index === -1) continue;
          const token = `${slots.length}`;
          html = html.slice(0, index) + token + html.slice(index + escaped.length);
          // 链接文字保留模型原本写的那一段，别用完整标题替换，
          // 否则正文会突然多出半截商品名；完整标题放 tooltip。
          slots.push({ asin: ref.asin, text: escaped, full: ref.escaped });
          linked.add(ref.asin);
          break;
        }
      });
    }

    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\r\n/g, "\n").replace(/\n/g, "<br>");
    html = html.replace(/(\d+)/g, (_, index) => {
      const ref = slots[Number(index)];
      return `<button type="button" class="product-link" data-asin="${escapeHtml(ref.asin)}" title="在商品库中查看：${ref.full}">${ref.text}</button>`;
    });
    return { html, linked };
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function removeEmpty() {
    const empty = inner.querySelector(".empty");
    if (empty) empty.remove();
  }

  function showEmpty() {
    removeEmpty();
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.innerHTML = `
      <div class="empty-mark">${ICON}</div>
      <h2>从一句需求开始</h2>
      <p>告诉我预算、使用场景，或者想对比的型号。我会检索真实商品库，
         只基于查到的资料回答，并记住你的长期偏好。</p>
      <div class="empty-chips"></div>`;
    const chips = empty.querySelector(".empty-chips");
    EXAMPLES.forEach((item) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "empty-chip";
      btn.textContent = item.label;
      btn.addEventListener("click", () => send(item.query));
      chips.appendChild(btn);
    });
    inner.appendChild(empty);
  }

  function addMessage(role, text, meta) {
    removeEmpty();
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}`;

    if (role !== "user") {
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.innerHTML = role === "error" ? "!" : ICON;
      wrap.appendChild(avatar);
    }

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const answer = document.createElement("div");
    answer.className = "answer";
    const rendered = formatAnswer(text, meta && meta.products);
    answer.innerHTML = rendered.html;
    bubble.appendChild(answer);

    if (role === "assistant" && meta) {
      addProductRefs(bubble, meta.products, rendered.linked);
      addMeta(bubble, meta);
    }

    wrap.appendChild(bubble);
    inner.appendChild(wrap);
    scrollToBottom();
  }

  function shortTitle(title, fallback) {
    const text = String(title || fallback || "").trim();
    const comma = text.indexOf(",");
    const head = comma > 12 ? text.slice(0, comma) : text;
    return head.length > 42 ? `${head.slice(0, 42)}…` : head;
  }

  function addProductRefs(bubble, products, linked) {
    if (!Array.isArray(products) || !products.length) return;
    // 正文里的标题没精确命中（模型缩写/改写）时，给一排兜底入口，
    // 保证导购提到的商品仍然能点进商品库。
    const missing = products.filter(
      (item) => item && item.asin && (!linked || !linked.has(item.asin))
    );
    if (!missing.length) return;

    const row = el("div", "product-refs");
    row.appendChild(el("span", "product-refs-label", "相关商品"));
    missing.forEach((item) => {
      const btn = el("button", "product-ref", shortTitle(item.title, item.asin));
      btn.type = "button";
      btn.dataset.asin = item.asin;
      btn.title = item.title || item.asin;
      row.appendChild(btn);
    });
    bubble.appendChild(row);
  }

  function addMeta(bubble, meta) {
    const metaRow = document.createElement("div");
    metaRow.className = "meta";

    if (meta.stopped) {
      const pill = document.createElement("span");
      pill.className = "pill stopped";
      pill.textContent = "已停止";
      metaRow.appendChild(pill);
    }
    if (meta.intent) {
      const pill = document.createElement("span");
      pill.className = "pill intent";
      pill.textContent = INTENT_LABEL[meta.intent] || meta.intent;
      metaRow.appendChild(pill);
    }
    if (meta.model) {
      const pill = document.createElement("span");
      pill.className = "pill model-pill";
      pill.textContent = MODEL_LABEL[meta.model] || meta.model;
      metaRow.appendChild(pill);
    }
    if (meta.steps !== undefined && meta.steps !== null) {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = `${meta.steps} 步工具调用`;
      metaRow.appendChild(pill);
    }
    if (meta.latency_ms) {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = `${(meta.latency_ms / 1000).toFixed(1)}s`;
      metaRow.appendChild(pill);
    }
    if (metaRow.childNodes.length) bubble.appendChild(metaRow);

    if (state.showTrace && Array.isArray(meta.trace) && meta.trace.length) {
      const details = document.createElement("details");
      details.className = "trace";
      const summary = document.createElement("summary");
      summary.textContent = "决策轨迹";
      const pre = document.createElement("pre");
      pre.textContent = meta.trace.join("\n");
      details.append(summary, pre);
      bubble.appendChild(details);
    }
  }

  function addStreamingMessage() {
    removeEmpty();
    const wrap = document.createElement("div");
    wrap.className = "msg assistant streaming";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.innerHTML = ICON;

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const answer = document.createElement("div");
    answer.className = "answer";
    bubble.appendChild(answer);
    wrap.append(avatar, bubble);
    inner.appendChild(wrap);
    scrollToBottom();

    let raw = "";
    let frame = 0;
    const paint = () => {
      frame = 0;
      answer.innerHTML = formatAnswer(raw).html;
      scrollToBottom();
    };

    return {
      get text() {
        return raw;
      },
      append(text) {
        raw += text;
        if (!frame) frame = requestAnimationFrame(paint);
      },
      finish(text, meta) {
        if (frame) cancelAnimationFrame(frame);
        raw = text || raw;
        const rendered = formatAnswer(raw, meta && meta.products);
        answer.innerHTML = rendered.html;
        wrap.classList.remove("streaming");
        if (meta) addProductRefs(bubble, meta.products, rendered.linked);
        addMeta(bubble, meta);
        scrollToBottom();
      },
      remove() {
        if (frame) cancelAnimationFrame(frame);
        wrap.remove();
      },
    };
  }

  function renderAll() {
    inner.innerHTML = "";
    if (!state.log.length) {
      showEmpty();
      return;
    }
    state.log.forEach((item) => addMessage(item.role, item.text, item.meta));
    scrollToBottom();
  }

  function addTyping() {
    removeEmpty();
    const wrap = document.createElement("div");
    wrap.className = "msg assistant typing";
    wrap.innerHTML = `
      <div class="avatar">${ICON}</div>
      <div class="bubble">
        <span class="dots"><span></span><span></span><span></span></span>
        <span class="typing-text">正在检索商品…</span>
      </div>`;
    inner.appendChild(wrap);
    scrollToBottom();

    const started = Date.now();
    const label = wrap.querySelector(".typing-text");
    let step = "正在检索商品…";
    let secs = 0;

    const paint = () => {
      label.textContent = `${step}${secs ? ` · ${secs}s` : ""}`;
    };
    const timer = setInterval(() => {
      secs = Math.round((Date.now() - started) / 1000);
      paint();
    }, 500);
    paint();

    return {
      setStep(line) {
        const text = String(line || "").trim();
        if (!text) return;
        step = text.length > 52 ? `${text.slice(0, 52)}…` : text;
        paint();
      },
      remove() {
        clearInterval(timer);
        wrap.remove();
      },
    };
  }

  // ---------- api ----------

  async function api(path, options) {
    const res = await fetch(path, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail ? String(data.detail) : `HTTP ${res.status}`);
    }
    return data;
  }

  // ---------- 模型选择 ----------

  const MODEL_LABEL = { deepseek: "DeepSeek", qwen: "Qwen 本地" };
  let modelCatalog = null;
  let modelTimer = 0;

  function modelInfo(id) {
    if (!modelCatalog || !Array.isArray(modelCatalog.models)) return null;
    return modelCatalog.models.find((item) => item.id === id) || null;
  }

  function renderModelOptions() {
    const qwen = modelInfo("qwen");
    const deepseek = modelInfo("deepseek");
    const qwenOpt = modelSelect.querySelector('option[value="qwen"]');
    const dsOpt = modelSelect.querySelector('option[value="deepseek"]');
    if (dsOpt) dsOpt.textContent = "DeepSeek（云端）";
    if (qwenOpt) {
      if (!qwen || qwen.available) {
        qwenOpt.textContent = "Qwen（本地 · 速度较慢）";
      } else {
        const checks = Array.isArray(qwen.checks) ? qwen.checks : [];
        const tunnel = checks.find((item) => item.id === "tunnel");
        const model = checks.find((item) => item.id === "model");
        let reason = "不可用";
        if (tunnel && tunnel.ok === false) reason = "隧道未连通";
        else if (model && model.ok === false) reason = "模型未就绪";
        qwenOpt.textContent = `Qwen（本地 · ${reason}）`;
      }
    }
    if (deepseek && !deepseek.available) {
      if (dsOpt) dsOpt.textContent = "DeepSeek（云端 · 不可用）";
    }
    modelSelect.value = state.model;
  }

  function renderModelStatus() {
    const info = modelInfo(state.model);
    if (!info) {
      modelStatusEl.dataset.state = "checking";
      modelStatusText.textContent = "正在检测模型…";
      return;
    }

    if (state.model === "deepseek") {
      modelStatusEl.dataset.state = info.available ? "ok" : "err";
      modelStatusText.textContent = info.available
        ? `云端 · ${info.model}`
        : `不可用：${info.detail}`;
      return;
    }

    modelStatusEl.dataset.state = info.available ? "ok" : "err";
    const checks = Array.isArray(info.checks) ? info.checks : [];
    const tunnel = checks.find((item) => item.id === "tunnel");
    const model = checks.find((item) => item.id === "model");
    const mark = (check) => {
      if (!check) return "";
      if (check.ok === true) return "✓";
      if (check.ok === false) return "✗";
      return "?";
    };
    const stages = [];
    if (tunnel) stages.push(`隧道 ${mark(tunnel)}`);
    if (model) stages.push(`模型 ${mark(model)}`);
    const stageText = stages.length ? `${stages.join(" · ")} · ` : "";
    modelStatusEl.title = checks.length
      ? checks.map((check) => `${check.label}：${check.detail}`).join("；")
      : info.detail;

    if (!info.available) {
      modelStatusText.textContent = `本地 · ${stageText}不可用：${info.detail}`;
      return;
    }
    const latency = info.latency_ms != null ? ` · ${info.latency_ms}ms` : "";
    modelStatusText.textContent = `本地 · ${stageText}${info.model}${latency}`;
  }

  async function refreshModels(force) {
    if (force) modelRefreshBtn.classList.add("is-busy");
    try {
      const data = await api(`/models${force ? "?refresh=1" : ""}`);
      modelCatalog = data;
      const valid = data.models.some((item) => item.id === state.model);
      if (!valid) {
        state.model = data.default || "deepseek";
        writeStorage("shopagent.model", state.model);
      }
      renderModelOptions();
      renderModelStatus();
    } catch (err) {
      modelCatalog = null;
      modelStatusEl.dataset.state = "err";
      modelStatusText.textContent = `检测失败：${err.message}`;
    } finally {
      modelRefreshBtn.classList.remove("is-busy");
    }
  }

  modelSelect.addEventListener("change", () => {
    state.model = modelSelect.value;
    writeStorage("shopagent.model", state.model);
    renderModelStatus();
    if (state.model === "qwen") refreshModels(false);
    inputEl.focus();
  });

  modelRefreshBtn.addEventListener("click", () => refreshModels(true));

  // Qwen 是本地服务，随时可能启动/关闭；选中时每 20s 轻量复查一次
  modelTimer = window.setInterval(() => {
    if (!document.hidden && state.model === "qwen" && !state.sending) {
      refreshModels(false);
    }
  }, 20000);

  async function streamChat(payload, onEvent, signal) {
    const res = await fetch("/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(payload),
      signal,
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail ? String(data.detail) : `HTTP ${res.status}`);
    }

    // 极老的浏览器没有 ReadableStream：退回一次性 JSON，行为不变
    if (!res.body) {
      const data = await res.json();
      onEvent("done", data);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let streamError = null;

    const dispatch = (block) => {
      let event = "message";
      const dataLines = [];
      block.split(/\r?\n/).forEach((line) => {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      });
      if (!dataLines.length) return;
      let data;
      try {
        data = JSON.parse(dataLines.join("\n"));
      } catch {
        return;
      }
      if (event === "error") {
        streamError = new Error(data.message || "生成失败");
      } else {
        onEvent(event, data);
      }
    };

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        while (true) {
          const match = buffer.match(/\r?\n\r?\n/);
          if (!match || match.index === undefined) break;
          const block = buffer.slice(0, match.index);
          buffer = buffer.slice(match.index + match[0].length);
          dispatch(block);
        }
        if (streamError) break;
      }
      buffer += decoder.decode();
      if (buffer.trim()) dispatch(buffer);
      if (streamError) throw streamError;
    } finally {
      try {
        await reader.cancel();
      } catch {
        /* 读完了/已断开都无所谓 */
      }
    }
  }

  function setSending(sending) {
    state.sending = sending;
    sendBtn.disabled = sending;
    inputEl.disabled = sending;
    modelSelect.disabled = sending;
    modelRefreshBtn.disabled = sending;
    composerEl.classList.toggle("is-sending", sending);
  }

  function stopGeneration() {
    if (!activeAbort || activeAbort.signal.aborted) return;
    stopRequested = true;
    activeAbort.abort();
  }

  async function send(rawText) {
    const text = (rawText || "").trim();
    if (!text || state.sending) return;

    if (state.model === "qwen") {
      const info = modelInfo("qwen");
      if (info && !info.available) {
        const message =
          `本地 Qwen 当前不可用：${info.detail}。` +
          "启动本地服务后点 ↻ 重新检测，或切回 DeepSeek。";
        state.log.push({ role: "error", text: message });
        addMessage("error", message);
        return;
      }
    }

    setSending(true);
    inputEl.value = "";
    inputEl.style.height = "auto";

    const controller = new AbortController();
    activeAbort = controller;
    stopRequested = false;

    state.log.push({ role: "user", text });
    addMessage("user", text);

    const typing = addTyping();
    const started = performance.now();
    let bubble = null;
    let final = null;

    try {
      await streamChat(
        {
          message: text,
          session_id: state.sessionId,
          use_history: true,
          model: state.model,
        },
        (event, data) => {
          if (event === "step") {
            typing.setStep(data.line);
          } else if (event === "token") {
            if (!bubble) {
              typing.remove();
              bubble = addStreamingMessage();
            }
            bubble.append(data.text || "");
          } else if (event === "done") {
            final = data;
          }
        },
        controller.signal
      );

      typing.remove();
      if (!final) throw new Error("连接中断，没有收到完整回答");

      const answer = final.answer || (bubble ? bubble.text : "") || "（模型没有返回内容，请重试）";
      const meta = {
        intent: final.intent,
        steps: final.steps,
        model: final.model || state.model,
        latency_ms: final.latency_ms || Math.round(performance.now() - started),
        trace: final.trace,
        products: Array.isArray(final.products) ? final.products : [],
      };

      if (bubble) {
        bubble.finish(answer, meta);
      } else {
        addMessage("assistant", answer, meta);
      }
      state.log.push({ role: "assistant", text: answer, meta });
      refreshProfile();
      // 服务端这时才把会话落盘，刷新历史列表让新会话出现/时间更新
      loadSessions();
    } catch (err) {
      typing.remove();
      if (stopRequested) {
        // 保留已经生成的部分并标注「已停止」；半截答案服务端不会落盘
        const partial = bubble ? bubble.text.trim() : "";
        const meta = { stopped: true, model: state.model };
        const shown = partial || "（已停止生成）";
        if (bubble) {
          bubble.finish(shown, meta);
        } else {
          addMessage("assistant", shown, meta);
        }
        state.log.push({ role: "assistant", text: shown, meta });
        return;
      }
      if (bubble) bubble.remove();
      const message = `请求失败：${err.message}。请确认服务已启动或稍后重试。`;
      state.log.push({ role: "error", text: message });
      addMessage("error", message);
    } finally {
      activeAbort = null;
      stopRequested = false;
      setSending(false);
      inputEl.focus();
    }
  }

  // ---------- 商品库页签 ----------

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function money(value) {
    return `$${moneyFmt.format(Number(value))}`;
  }

  function pct(part, total) {
    return total ? `${((part / total) * 100).toFixed(1)}%` : "0%";
  }

  function initials(name) {
    const parts = String(name || "?").trim().split(/\s+/).filter(Boolean);
    if (!parts.length) return "?";
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
  }

  function statCard(label, value, note) {
    const card = el("div", "stat-card");
    card.append(el("span", "stat-label", label), el("strong", "", value), el("small", "", note));
    return card;
  }

  function renderStats(data) {
    catalogStatsEl.innerHTML = "";
    catalogStatsEl.append(
      statCard("商品总数", nf.format(data.total), "Sports & Outdoors"),
      statCard("有标价", nf.format(data.priced), `${pct(data.priced, data.total)} 有价格`),
      statCard("品牌数", nf.format(data.brands), "含大量长尾品牌"),
      statCard(
        "中位价",
        data.median_price == null ? "—" : money(data.median_price),
        data.price_min == null ? "暂无价格数据" : `区间 ${money(data.price_min)} – ${money(data.price_max)}`
      )
    );

    priceNote.textContent = `${pct(data.priced, data.total)} 的商品有标价`;
    brandNote.textContent = `共 ${nf.format(data.brands)} 个品牌，展示 Top 8`;

    const max = Math.max(...data.histogram.map((bin) => bin.count), data.unpriced, 1);
    priceBarsEl.innerHTML = "";
    data.histogram.forEach((bin) => appendBar(priceBarsEl, bin.label, bin.count, max));
    appendBar(priceBarsEl, "未标价", data.unpriced, max, true);

    const brandMax = Math.max(...data.top_brands.map((item) => item.count), 1);
    brandBarsEl.innerHTML = "";
    data.top_brands.forEach((item) => appendBar(brandBarsEl, item.brand, item.count, brandMax));

    catalogSource.textContent = `数据：Amazon 商品元数据 · ${nf.format(data.total)} 条`;
  }

  function appendBar(container, label, count, max, muted = false) {
    const row = el("div", muted ? "bar-row is-muted" : "bar-row");
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.title = `${label}：${nf.format(count)} 条`;
    track.appendChild(fill);
    row.append(
      el("span", "bar-label", label),
      track,
      el("span", "bar-value", nf.format(count))
    );
    container.appendChild(row);
    // 先以 0 宽度插入 DOM，再在下一帧给目标宽度，条形才有增长动画
    requestAnimationFrame(() => {
      fill.style.width = `${(count / max) * 100}%`;
    });
  }

  async function loadCatalogStats() {
    try {
      const data = await api("/catalog/stats");
      catalogState.statsLoaded = true;
      renderStats(data);
    } catch (err) {
      catalogStatsEl.innerHTML = "";
      catalogStatsEl.appendChild(el("p", "catalog-empty", `概览加载失败：${err.message}`));
    }
  }

  function catalogParams() {
    const params = new URLSearchParams();
    params.set("page", String(catalogState.page));
    params.set("page_size", "12");
    if (catalogState.query) params.set("q", catalogState.query);
    if (catalogState.sort !== "default") params.set("sort", catalogState.sort);

    const price = catalogState.price;
    if (price === "priced") params.set("priced", "yes");
    else if (price === "unpriced") params.set("priced", "no");
    else if (price === "0-25") {
      params.set("price_min", "0");
      params.set("price_max", "25");
    } else if (price === "25-50") {
      params.set("price_min", "25");
      params.set("price_max", "50");
    } else if (price === "50-100") {
      params.set("price_min", "50");
      params.set("price_max", "100");
    } else if (price === "100-") {
      params.set("price_min", "100");
    }
    return params;
  }

  async function loadCatalogProducts() {
    const requestId = ++catalogState.requestId;
    catalogCount.textContent = "正在加载商品…";
    catalogPager.innerHTML = "";
    productGrid.innerHTML = "";
    productGrid.appendChild(el("p", "catalog-empty", "正在加载商品…"));

    try {
      const data = await api(`/catalog/products?${catalogParams().toString()}`);
      if (requestId !== catalogState.requestId) return;
      catalogState.loaded = true;
      renderProducts(data);
    } catch (err) {
      if (requestId !== catalogState.requestId) return;
      catalogCount.textContent = "";
      productGrid.innerHTML = "";
      const box = el("div", "catalog-empty");
      box.appendChild(document.createTextNode(`加载失败：${err.message} `));
      const retry = el("button", "page-btn", "重试");
      retry.type = "button";
      retry.addEventListener("click", loadCatalogProducts);
      box.appendChild(retry);
      productGrid.appendChild(box);
    }
  }

  function renderProducts(data) {
    productGrid.innerHTML = "";
    catalogPager.innerHTML = "";

    if (!data.items.length) {
      catalogCount.textContent = "没有匹配的商品";
      productGrid.appendChild(
        el("p", "catalog-empty", "没有匹配的商品。试试减少关键词、放宽价格，或回到「对话」让 Agent 帮你找。")
      );
      return;
    }

    catalogCount.textContent =
      `找到 ${nf.format(data.total)} 条 · 第 ${data.page} / ${data.pages} 页`;

    const frag = document.createDocumentFragment();
    data.items.forEach((item) => frag.appendChild(productCard(item)));
    productGrid.appendChild(frag);
    renderPager(data);
  }

  function productCard(item) {
    const card = el("article", "product-card");
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `查看 ${item.title || item.asin} 的完整信息`);
    card.addEventListener("click", () => openProduct(item.asin));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openProduct(item.asin);
      }
    });

    const top = el("div", "product-top");
    const badge = el("span", "brand-badge", initials(item.brand));
    badge.setAttribute("aria-hidden", "true");
    const brand = el("span", "product-brand", item.brand || "未知品牌");
    brand.title = item.brand || "";
    const unknownPrice = item.price === null || item.price === undefined;
    const price = el(
      "span",
      unknownPrice ? "product-price is-unknown" : "product-price",
      unknownPrice ? "未标价" : money(item.price)
    );
    top.append(badge, brand, price);

    const title = el("h3", "", item.title);
    title.title = item.title;
    // snippet 是「关键词命中在描述/卖点里」时截出的上下文，优先于普通描述；
    // 这个数据集里不少 description 只有一个词（例如 "punch"），太短就改用 features。
    const description = String(item.description || "").trim();
    const descText =
      item.snippet ||
      (description.length >= 24 ? item.description : item.features || item.description) ||
      "暂无描述";
    const desc = el(
      "p",
      item.snippet ? "product-desc is-match" : "product-desc",
      descText
    );

    const foot = el("div", "product-foot");
    foot.append(el("code", "", item.asin), el("span", "product-more", "查看全部信息 →"));

    card.append(top, title, desc, foot);
    return card;
  }

  function openProductFromChat(asin) {
    setView("catalog");
    openProduct(asin);
  }

  async function openProduct(asin) {
    if (!asin) return;
    const requestId = ++detailRequestId;

    if (typeof productModal.showModal === "function") {
      if (!productModal.open) productModal.showModal();
    } else {
      productModal.setAttribute("open", "");
    }
    document.body.classList.add("modal-open");
    modalKicker.textContent = "商品详情";
    modalTitle.textContent = "正在加载…";
    modalBody.innerHTML = "";
    modalBody.scrollTop = 0;
    modalBody.appendChild(el("p", "detail-loading", "正在加载商品信息…"));

    try {
      const product = await api(`/catalog/products/${encodeURIComponent(asin)}`);
      if (requestId !== detailRequestId) return;
      renderProductDetail(product);
    } catch (err) {
      if (requestId !== detailRequestId) return;
      modalTitle.textContent = "商品详情";
      modalBody.innerHTML = "";
      modalBody.appendChild(el("p", "detail-error", `加载失败：${err.message}`));
    }
  }

  function formatFeatureText(text) {
    return String(text || "").replace(/\s*【/g, "\n【").trim();
  }

  function addDetailSection(title, text) {
    const section = el("section", "detail-section");
    section.append(el("h3", "", title), el("p", "detail-text", text));
    modalBody.appendChild(section);
  }

  function renderProductDetail(product) {
    modalKicker.textContent = product.brand || "商品详情";
    modalTitle.textContent = product.title || product.asin || "商品详情";
    modalBody.innerHTML = "";

    const hero = el("div", "detail-hero");
    const badge = el("span", "brand-badge", initials(product.brand));
    badge.setAttribute("aria-hidden", "true");
    const brand = el("div", "detail-brand");
    brand.append(
      el("span", "detail-brand-name", product.brand || "未知品牌"),
      el("span", "detail-asin", `ASIN ${product.asin || "—"}`)
    );
    const unknownPrice = product.price === null || product.price === undefined;
    const price = el(
      "div",
      unknownPrice ? "detail-price is-unknown" : "detail-price",
      unknownPrice ? "未标价，需自行确认" : money(product.price)
    );
    hero.append(badge, brand, price);

    const tags = el("div", "detail-tags");
    if (product.category) tags.appendChild(el("span", "detail-tag", product.category));
    if (unknownPrice) tags.appendChild(el("span", "detail-tag is-warn", "价格缺失"));

    const copy = el("button", "detail-copy", "复制 ASIN");
    copy.type = "button";
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(String(product.asin || ""));
        copy.textContent = "已复制";
      } catch {
        copy.textContent = "复制失败";
      }
    });
    tags.appendChild(copy);

    modalBody.append(hero, tags);
    addDetailSection("商品描述", product.description || "暂无描述");
    addDetailSection("卖点 / Features", formatFeatureText(product.features) || "暂无卖点");

    const raw = document.createElement("details");
    raw.className = "detail-raw";
    const summary = document.createElement("summary");
    summary.textContent = "原始记录（JSON）";
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(product, null, 2);
    raw.append(summary, pre);
    modalBody.appendChild(raw);
  }

  function renderPager(data) {
    if (data.pages <= 1) return;

    const prev = el("button", "page-btn", "上一页");
    prev.type = "button";
    prev.disabled = data.page <= 1;
    prev.addEventListener("click", () => goToPage(data.page - 1));

    const next = el("button", "page-btn", "下一页");
    next.type = "button";
    next.disabled = data.page >= data.pages;
    next.addEventListener("click", () => goToPage(data.page + 1));

    catalogPager.append(
      prev,
      el("span", "page-label", `第 ${data.page} / ${data.pages} 页 · 共 ${nf.format(data.total)} 条`),
      next
    );
  }

  function goToPage(page) {
    if (page < 1 || page === catalogState.page) return;
    catalogState.page = page;
    loadCatalogProducts();
    document.querySelector(".catalog-browser").scrollIntoView({ block: "start", behavior: "smooth" });
  }

  function searchCatalog() {
    catalogState.query = catalogQ.value.trim();
    catalogState.page = 1;
    loadCatalogProducts();
  }

  let searchTimer = 0;
  catalogQ.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(searchCatalog, 250);
  });
  catalogQ.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      clearTimeout(searchTimer);
      searchCatalog();
    }
  });

  catalogPrice.addEventListener("change", () => {
    catalogState.price = catalogPrice.value;
    catalogState.page = 1;
    loadCatalogProducts();
  });

  catalogSort.addEventListener("change", () => {
    catalogState.sort = catalogSort.value;
    catalogState.page = 1;
    loadCatalogProducts();
  });

  function setView(view) {
    const isCatalog = view === "catalog";
    mainEl.classList.toggle("is-catalog", isCatalog);
    catalogEl.hidden = !isCatalog;
    tabChat.classList.toggle("is-active", !isCatalog);
    tabCatalog.classList.toggle("is-active", isCatalog);
    tabChat.setAttribute("aria-selected", String(!isCatalog));
    tabCatalog.setAttribute("aria-selected", String(isCatalog));

    const hash = isCatalog ? "#catalog" : "#chat";
    if (location.hash !== hash) history.replaceState(null, "", hash);

    if (isCatalog) {
      viewTitle.textContent = "商品库";
      subtitle.textContent =
        "Amazon 运动户外（Sports & Outdoors）10 万条商品 · 概览、价格分布与快速检索";
      if (!catalogState.statsLoaded) loadCatalogStats();
      if (!catalogState.loaded) loadCatalogProducts();
    } else {
      viewTitle.textContent = "购物助手";
      subtitle.textContent =
        "Amazon 运动户外（Sports & Outdoors）10 万条商品 · 说需求，我来检索、比较、给建议";
      inputEl.focus();
    }
  }

  tabChat.addEventListener("click", () => setView("chat"));
  tabCatalog.addEventListener("click", () => setView("catalog"));

  inner.addEventListener("click", (event) => {
    const trigger = event.target.closest(".product-link, .product-ref");
    if (!trigger) return;
    const asin = trigger.dataset.asin;
    if (asin) openProductFromChat(asin);
  });

  modalClose.addEventListener("click", () => productModal.close());
  productModal.addEventListener("click", (event) => {
    if (event.target === productModal) productModal.close();
  });
  productModal.addEventListener("close", () => {
    detailRequestId += 1;
    document.body.classList.remove("modal-open");
  });

  // ---------- sidebar ----------

  function renderExamples() {
    examplesEl.innerHTML = "";
    EXAMPLES.forEach((item) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "example";
      btn.textContent = item.label;
      btn.addEventListener("click", () => send(item.query));
      examplesEl.appendChild(btn);
    });
  }

  function profileChips(profile) {
    const chips = [];
    if (profile.budget_max) chips.push(["预算", `$${profile.budget_max}`]);
    if (profile.preferred_brand) chips.push(["品牌", String(profile.preferred_brand)]);
    if (profile.skin_type) chips.push(["偏好", String(profile.skin_type)]);
    if (Array.isArray(profile.avoid) && profile.avoid.length) {
      chips.push(["避开", profile.avoid.join("、")]);
    }
    if (profile.notes) chips.push(["备注", String(profile.notes)]);
    return chips;
  }

  function renderProfile(profile) {
    profileEl.innerHTML = "";
    const chips = profileChips(profile || {});
    if (!chips.length) {
      const p = document.createElement("p");
      p.className = "muted";
      p.textContent = "还没有长期偏好。聊几句，我会记住你的预算和偏好。";
      profileEl.appendChild(p);
      return;
    }
    chips.forEach(([key, value]) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      const label = document.createElement("b");
      label.textContent = key;
      chip.append(label, document.createTextNode(value));
      profileEl.appendChild(chip);
    });
  }

  async function refreshProfile() {
    try {
      const data = await api("/profile");
      renderProfile(data.profile);
    } catch {
      renderProfile({});
    }
  }

  // 检索漏斗：召回池固定 50（retrieval 的 candidate_k 默认值），输出固定 5；
  // 中间精排候选来自 /health，拖动滑块会 POST /config/rerank_candidates 热改，
  // 只改服务内存，新请求立刻生效，进程重启后回到 .env 里的默认值。
  const RECALL_POOL = 50;
  const OUTPUT_K = 5;
  const CONFIG_NOTE = "拖动即时生效，重启后恢复 .env 默认值";

  let appliedCandidates = null;
  let configRequest = 0;
  let noteTimer = 0;

  function funnelWidth(count) {
    const ratio = Math.min(1, Math.max(0, count / RECALL_POOL));
    return `${(ratio * 100).toFixed(1)}%`;
  }

  function previewCandidates(value) {
    funnelRerankValue.textContent = nf.format(value);
    funnelRerank.style.width = funnelWidth(value);
    configEditValue.textContent = nf.format(value);
  }

  function setConfigNote(text, kind) {
    window.clearTimeout(noteTimer);
    configNoteText.textContent = text;
    configNoteEl.classList.toggle("is-ok", kind === "ok");
    configNoteEl.classList.toggle("is-err", kind === "err");
    if (kind) {
      noteTimer = window.setTimeout(() => {
        configNoteText.textContent = CONFIG_NOTE;
        configNoteEl.classList.remove("is-ok", "is-err");
      }, 2600);
    }
  }

  function renderConfig(health) {
    const candidates = Number(health.rerank_candidates);
    const mode = health.mode || "—";
    // 只有 hybrid_rerank 这条链路里精排候选数才有意义
    const onRerank = mode.includes("rerank") && Number.isFinite(candidates);

    configMode.textContent = mode;
    configPanel.classList.toggle("is-off", !onRerank);
    configRange.disabled = !onRerank;
    funnelRecall.style.width = funnelWidth(RECALL_POOL);
    funnelOutput.style.width = funnelWidth(OUTPUT_K);

    if (onRerank) {
      appliedCandidates = candidates;
      configRange.value = String(candidates);
      previewCandidates(candidates);
    } else {
      appliedCandidates = null;
      funnelRerank.style.width = "0%";
      funnelRerankValue.textContent = "—";
      configEditValue.textContent = "—";
    }
    setConfigNote(onRerank ? CONFIG_NOTE : "当前检索模式未启用精排");
  }

  configRange.addEventListener("input", () => {
    previewCandidates(Number(configRange.value));
  });

  // range 的 change 在松手（或键盘松开）时触发，一次拖动只发一个请求
  configRange.addEventListener("change", async () => {
    const value = Number(configRange.value);
    const requestId = ++configRequest;
    try {
      const data = await api("/config/rerank_candidates", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value }),
      });
      if (requestId !== configRequest) return;
      appliedCandidates = Number(data.rerank_candidates);
      configRange.value = String(appliedCandidates);
      previewCandidates(appliedCandidates);
      setConfigNote(`已应用：精排取前 ${appliedCandidates} 条`, "ok");
    } catch (err) {
      if (requestId !== configRequest) return;
      if (appliedCandidates != null) {
        configRange.value = String(appliedCandidates);
        previewCandidates(appliedCandidates);
      }
      setConfigNote(`修改失败：${err.message}`, "err");
    }
  });

  async function refreshHealth() {
    try {
      const health = await api("/health");
      statusEl.classList.add("ok");
      statusEl.classList.remove("bad");
      statusText.textContent = `${health.store} · ${health.mode} · ${health.products} 商品`;
      renderConfig(health);
    } catch {
      statusEl.classList.add("bad");
      statusEl.classList.remove("ok");
      statusText.textContent = "服务未连接";
      configMode.textContent = "—";
      funnelRerankValue.textContent = "—";
      configEditValue.textContent = "—";
      configRange.disabled = true;
    }
  }

  // ---------- 历史会话 ----------

  let sessionsCache = [];

  function sessionTime(iso) {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    const pad = (n) => String(n).padStart(2, "0");
    const hm = `${pad(date.getHours())}:${pad(date.getMinutes())}`;
    if (date.toDateString() === new Date().toDateString()) return `今天 ${hm}`;
    return `${date.getMonth() + 1}-${pad(date.getDate())} ${hm}`;
  }

  function renderSessions(failed) {
    const scrollTop = sessionsEl.scrollTop;
    sessionsEl.innerHTML = "";
    sessionCount.textContent = sessionsCache.length ? `${sessionsCache.length} 条` : "";
    if (failed) {
      sessionsEl.appendChild(el("p", "muted", "历史会话加载失败"));
      return;
    }
    if (!sessionsCache.length) {
      sessionsEl.appendChild(el("p", "muted", "还没有历史会话"));
      return;
    }

    sessionsCache.forEach((item) => {
      const active = item.session_id === state.sessionId;
      const row = el("div", active ? "session-item is-active" : "session-item");

      const open = el("button", "session-open");
      open.type = "button";
      open.title = item.title;
      open.append(
        el("span", "session-title", item.title),
        el("span", "session-meta", `${sessionTime(item.updated_at)} · ${item.message_count} 条消息`)
      );
      open.addEventListener("click", () => openSession(item.session_id));

      const del = el("button", "session-del", "×");
      del.type = "button";
      del.title = "删除这条会话";
      del.setAttribute("aria-label", `删除会话：${item.title}`);
      del.addEventListener("click", (event) => {
        event.stopPropagation();
        deleteSession(item.session_id);
      });

      row.append(open, del);
      sessionsEl.appendChild(row);
    });
    // 每次发消息后都会重绘列表，别把用户滚到的位置弹回顶部
    sessionsEl.scrollTop = scrollTop;
  }

  async function loadSessions() {
    try {
      const data = await api("/sessions");
      sessionsCache = data.sessions || [];
      renderSessions(false);
    } catch {
      sessionsCache = [];
      renderSessions(true);
    }
  }

  async function openSession(sessionId) {
    if (state.sending || sessionId === state.sessionId) return;
    state.sessionId = sessionId;
    writeStorage("shopagent.session", sessionId);
    await loadHistory();
    renderSessions(false);
    inputEl.focus();
  }

  async function deleteSession(sessionId) {
    if (state.sending) return;
    if (!window.confirm("删除这条历史会话？删除后不能恢复。")) return;
    try {
      await api(`/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      if (sessionId === state.sessionId) await startNewSession();
      await loadSessions();
    } catch (err) {
      window.alert(`删除失败：${err.message}`);
    }
  }

  async function loadHistory() {
    try {
      const data = await api(`/sessions/${encodeURIComponent(state.sessionId)}`);
      const messages = data.messages || [];
      state.log = messages.map((message) => ({
        role: message.role === "user" ? "user" : "assistant",
        text: message.content,
        meta: message.meta,
      }));
    } catch {
      state.log = [];
    }
    renderAll();
  }

  async function startNewSession() {
    if (state.sending) return;
    // 只切到一个新的空会话，不再删除旧会话——历史要能被点回来
    state.sessionId = newSessionId();
    writeStorage("shopagent.session", state.sessionId);
    state.log = [];
    renderAll();
    renderSessions(false);
    inputEl.focus();
  }

  async function clearProfile() {
    if (!window.confirm("清空长期画像？预算、品牌等偏好会被删除。")) return;
    try {
      await api("/profile", { method: "DELETE" });
      refreshProfile();
    } catch (err) {
      window.alert(`清空失败：${err.message}`);
    }
  }

  // ---------- events ----------

  stopBtn.addEventListener("click", stopGeneration);

  composerEl.addEventListener("submit", (event) => {
    event.preventDefault();
    send(inputEl.value);
  });

  inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send(inputEl.value);
    }
  });

  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
  });

  $("#new-chat").addEventListener("click", startNewSession);
  $("#clear-profile").addEventListener("click", clearProfile);

  traceToggle.checked = state.showTrace;
  traceToggle.addEventListener("change", () => {
    state.showTrace = traceToggle.checked;
    writeStorage("shopagent.trace", state.showTrace ? "1" : "0");
    // 流式输出期间不要整页重绘，否则会把正在增长的答案清掉
    if (!state.sending) renderAll();
  });

  // ---------- boot ----------

  renderExamples();
  refreshHealth();
  refreshModels(false);
  refreshProfile();
  loadHistory();
  loadSessions();
  if (location.hash === "#catalog") setView("catalog");
  else inputEl.focus();
})();
