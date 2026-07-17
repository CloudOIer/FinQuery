const STORAGE_KEY = 'finquery-chat-conversations-v1';

const questionInput = document.querySelector('#question');
const form = document.querySelector('#query-form');
const submitButton = document.querySelector('#submit-button');
const messagesEl = document.querySelector('#messages');
const conversationListEl = document.querySelector('#conversation-list');
const newChatButton = document.querySelector('#new-chat');

let conversations = [];
let activeConversationId = null;

function createSessionId() {
  return `session-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function loadConversations() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    conversations = Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    conversations = [];
  }
  if (!conversations.length) {
    createConversation({ activate: true, persist: false });
  } else {
    activeConversationId = conversations[0].id;
  }
}

function saveConversations() {
  const compact = conversations.slice(0, 30).map((conversation) => ({
    ...conversation,
    messages: conversation.messages.slice(-40),
  }));
  localStorage.setItem(STORAGE_KEY, JSON.stringify(compact));
}

function getActiveConversation() {
  return conversations.find((conversation) => conversation.id === activeConversationId) || null;
}

function createConversation({ activate = true, persist = true } = {}) {
  const conversation = {
    id: createSessionId(),
    title: '新对话',
    updatedAt: Date.now(),
    messages: [],
  };
  conversations.unshift(conversation);
  if (activate) activeConversationId = conversation.id;
  if (persist) saveConversations();
  renderConversationList();
  renderMessages();
  questionInput.focus();
  return conversation;
}

function deleteConversation(id) {
  conversations = conversations.filter((conversation) => conversation.id !== id);
  if (!conversations.length) {
    createConversation({ activate: true });
    return;
  }
  if (activeConversationId === id) activeConversationId = conversations[0].id;
  saveConversations();
  renderConversationList();
  renderMessages();
}

function activateConversation(id) {
  activeConversationId = id;
  renderConversationList();
  renderMessages();
  questionInput.focus();
}

function updateConversationTitle(conversation, question) {
  if (conversation.title !== '新对话') return;
  const normalized = question.replace(/\s+/g, ' ').trim();
  conversation.title = normalized.length > 22 ? `${normalized.slice(0, 21)}…` : normalized;
}

function renderConversationList() {
  conversationListEl.innerHTML = conversations
    .map((conversation) => {
      const activeClass = conversation.id === activeConversationId ? ' active' : '';
      return `
        <div class="conversation-row${activeClass}">
          <button type="button" class="conversation-item" data-conversation-id="${escapeHtml(conversation.id)}">
            <span>${escapeHtml(conversation.title)}</span>
            <time>${escapeHtml(formatTime(conversation.updatedAt))}</time>
          </button>
          <button type="button" class="delete-chat" aria-label="删除对话" data-delete-id="${escapeHtml(conversation.id)}">×</button>
        </div>
      `;
    })
    .join('');
}

function renderMessages() {
  const conversation = getActiveConversation();
  if (!conversation || !conversation.messages.length) {
    messagesEl.innerHTML = `
      <div class="welcome-state">
        <h2>今天想查什么财报问题？</h2>
      </div>
    `;
    return;
  }

  messagesEl.innerHTML = conversation.messages.map(renderMessage).join('');
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderMessage(message) {
  const loadingClass = message.loading ? ' loading' : '';
  const chartImages = message.chartImages || (message.chartImage ? [message.chartImage] : []);
  const charts = chartImages.map(renderChartImage).join('');
  const sources = message.sources && message.sources.length ? renderSources(message.sources) : '';
  const trace = message.trace && message.trace.length ? renderTrace(message.trace) : '';
  return `
    <article class="message ${escapeHtml(message.role)}${loadingClass}">
      <div class="message-bubble">
        <div class="message-text">${escapeHtml(message.text).replaceAll('\n', '<br>')}</div>
        ${trace}
        ${charts}
        ${sources}
      </div>
    </article>
  `;
}

function renderTrace(trace) {
  const items = trace.map((step) => {
    const statusClass = step.status === 'error' ? ' trace-error' : '';
    const args = step.arguments && Object.keys(step.arguments).length ? JSON.stringify(step.arguments) : '';
    return `
      <li class="trace-step${statusClass}">
        <span class="trace-tool">${escapeHtml(String(step.step))}. ${escapeHtml(step.tool || '')}</span>
        ${args ? `<code class="trace-args">${escapeHtml(args.length > 160 ? args.slice(0, 159) + '…' : args)}</code>` : ''}
        <span class="trace-summary">${escapeHtml(step.summary || '')}${step.elapsed_seconds ? `(${escapeHtml(String(step.elapsed_seconds))}s)` : ''}</span>
      </li>
    `;
  }).join('');
  return `<details class="trace-list"><summary>执行过程(${trace.length} 步)</summary><ol>${items}</ol></details>`;
}

function renderChartImage(chartImage) {
  if (!chartImage.image_data_url) return '';
  const alt = chartImage.alt_text || chartImage.title || '统计图';
  return `
    <figure class="chart-image">
      <img src="${escapeHtml(chartImage.image_data_url)}" alt="${escapeHtml(alt)}" />
      ${chartImage.title ? `<figcaption>${escapeHtml(chartImage.title)}</figcaption>` : ''}
    </figure>
  `;
}

function renderSources(sources) {
  const items = sources.slice(0, 6).map((source, index) => {
    const meta = [source.org_name, source.publish_date, source.section_title].filter(Boolean).join(' · ');
    const snippet = source.snippet || source.text || '';
    return `
      <li>
        <span class="source-title">${index + 1}. ${escapeHtml(source.title || '研报来源')}</span>
        ${meta ? `<span class="source-meta">${escapeHtml(meta)}</span>` : ''}
        ${snippet ? `<span class="source-snippet">${escapeHtml(snippet)}</span>` : ''}
      </li>
    `;
  }).join('');
  return `<details class="source-list"><summary>参考来源</summary><ol>${items}</ol></details>`;
}

function addMessage(role, text, extra = {}) {
  let conversation = getActiveConversation();
  if (!conversation) conversation = createConversation({ activate: true });
  const message = {
    id: `message-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    role,
    text,
    createdAt: Date.now(),
    ...extra,
  };
  conversation.messages.push(message);
  conversation.updatedAt = Date.now();
  conversations = [conversation, ...conversations.filter((item) => item.id !== conversation.id)];
  activeConversationId = conversation.id;
  saveConversations();
  renderConversationList();
  renderMessages();
  return message.id;
}

function replaceMessage(messageId, patch) {
  const conversation = getActiveConversation();
  if (!conversation) return;
  conversation.messages = conversation.messages.map((message) => (message.id === messageId ? { ...message, ...patch } : message));
  conversation.updatedAt = Date.now();
  saveConversations();
  renderConversationList();
  renderMessages();
}

async function submitQuestion(event) {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    questionInput.focus();
    return;
  }

  let conversation = getActiveConversation();
  if (!conversation) conversation = createConversation({ activate: true });
  updateConversationTitle(conversation, question);
  addMessage('user', question);
  const pendingMessageId = addMessage('assistant', '正在查询...', { loading: true });
  questionInput.value = '';
  resizeInput();
  setSubmitting(true);

  try {
    const agentMode = Boolean(document.getElementById('agent-mode')?.checked);
    const response = await fetch('/analysis/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        use_llm: true,
        use_rag: true,
        use_vector: true,
        rag_top_k: 8,
        session_id: conversation.id,
        use_agent: agentMode,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }

    if (payload.status === 'clarification') {
      replaceMessage(pendingMessageId, {
        text: payload.clarification?.question || '需要补充信息后才能继续查询。',
        loading: false,
      });
      return;
    }

    replaceMessage(pendingMessageId, {
      text: payload.answer_text || '查询已完成，但暂时没有可展示的自然语言回答。',
      chartImages: payload.chart_images || (payload.chart_image ? [payload.chart_image] : []),
      sources: payload.sources || [],
      trace: payload.execution_trace || [],
      loading: false,
    });
  } catch (error) {
    replaceMessage(pendingMessageId, {
      text: `请求失败：${error.message}`,
      loading: false,
    });
  } finally {
    setSubmitting(false);
    questionInput.focus();
  }
}

function setSubmitting(isSubmitting) {
  submitButton.disabled = isSubmitting;
  questionInput.disabled = isSubmitting;
  submitButton.textContent = isSubmitting ? '发送中' : '发送';
}

function resizeInput() {
  questionInput.style.height = 'auto';
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 160)}px`;
}

function formatTime(timestamp) {
  return new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(timestamp));
}

function bindEvents() {
  form.addEventListener('submit', submitQuestion);
  newChatButton.addEventListener('click', () => createConversation({ activate: true }));
  questionInput.addEventListener('input', resizeInput);
  questionInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  conversationListEl.addEventListener('click', (event) => {
    const deleteButton = event.target.closest('[data-delete-id]');
    if (deleteButton) {
      deleteConversation(deleteButton.dataset.deleteId);
      return;
    }
    const item = event.target.closest('[data-conversation-id]');
    if (item) activateConversation(item.dataset.conversationId);
  });
}

loadConversations();
bindEvents();
renderConversationList();
renderMessages();
resizeInput();
