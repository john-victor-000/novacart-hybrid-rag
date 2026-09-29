import { useEffect, useRef, useState } from 'react'
import { BorderBeam } from 'border-beam'
import { BotAvatar } from 'bot-avatars'
import { MetalFx } from 'metal-fx'
import { ThinkingOrb } from 'thinking-orbs'
import './App.css'
import { ThemeToggle } from './components/ThemeToggle'
import { useTheme } from './hooks/useTheme'
import { ApiError, sendChatMessage } from './services/api'

const STARTER_QUESTIONS = [
  'What is the standard return window?',
  'What is the warranty of NCM-24?',
  'Which products have more than 12 months warranty?',
]

const CAPABILITIES = [
  'Dense Search',
  'BM25',
  'Hybrid RAG',
  'Reranking',
  'Citations',
]

const WELCOME_MESSAGE = {
  id: 'welcome',
  role: 'assistant',
  content:
    'Ask me about NovaCart products, delivery, returns, warranties, or order policies. I will answer from the available company sources.',
  sources: [],
}

function App() {
  const { theme, toggleTheme } = useTheme()
  const [messages, setMessages] = useState([WELCOME_MESSAGE])
  const [query, setQuery] = useState('')
  const [conversationId, setConversationId] = useState(() =>
    sessionStorage.getItem('novacart_conversation_id'),
  )
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const messagesRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    const messagesElement = messagesRef.current
    if (!messagesElement) return

    messagesElement.scrollTo({
      top: messagesElement.scrollHeight,
      behavior: messages.length > 1 ? 'smooth' : 'auto',
    })
  }, [messages, isLoading])

  useEffect(() => {
    if (conversationId) {
      sessionStorage.setItem('novacart_conversation_id', conversationId)
    }
  }, [conversationId])

  const submitMessage = async (text = query) => {
    const normalized = text.trim()
    if (!normalized || isLoading) return

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: normalized,
    }
    setMessages((current) => [...current, userMessage])
    setQuery('')
    setError('')
    setIsLoading(true)

    try {
      let response
      try {
        response = await sendChatMessage({
          query: normalized,
          conversationId,
        })
      } catch (requestError) {
        if (requestError instanceof ApiError && requestError.status === 404) {
          sessionStorage.removeItem('novacart_conversation_id')
          setConversationId(null)
          response = await sendChatMessage({
            query: normalized,
            conversationId: null,
          })
        } else {
          throw requestError
        }
      }

      setConversationId(response.conversation_id)
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
          route: response.route,
          metadata: response.metadata,
        },
      ])
    } catch (requestError) {
      const message =
        requestError instanceof ApiError
          ? requestError.message
          : 'Something went wrong while contacting NovaCart.'
      setError(message)
    } finally {
      setIsLoading(false)
      textareaRef.current?.focus()
    }
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submitMessage()
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-avatar">
            <BotAvatar
              type="droid"
              face="eyes"
              state={isLoading ? 'working' : 'default'}
              size={44}
              speed={0.8}
              seed={0.36}
              shading="plastic"
              whirl={0.45}
              jumpEvery={14}
              theme={theme}
              aria-label={
                isLoading
                  ? 'NovaCart droid assistant working'
                  : 'NovaCart droid assistant ready'
              }
            />
          </div>
          <div>
            <p className="eyebrow">NOVACART KNOWLEDGE</p>
            <h1>AI Assistant</h1>
          </div>
        </div>

        <div className="topbar-actions">
          <div
            className={
              'system-status' + (isLoading ? ' system-status--active' : '')
            }
          >
            <span className="status-dot" aria-hidden="true" />
            <span>{isLoading ? 'Searching sources' : 'Ready'}</span>
          </div>
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>
      </header>

      <main className="chat-layout">
        <section className="intro-panel" aria-labelledby="intro-title">
          <div className="hero-kicker">
            <span className="hero-kicker__dot" aria-hidden="true" />
            HYBRID INTELLIGENCE
          </div>
          <h2 id="intro-title">
            Answers backed by <span>NovaCart sources.</span>
          </h2>
          <p>
            Product facts, policies, and operational answers grounded in dense
            search, exact matching, reranking, and structured data.
          </p>
          <div className="capability-list" aria-label="Retrieval capabilities">
            {CAPABILITIES.map((capability) => (
              <span key={capability}>{capability}</span>
            ))}
          </div>
          <div className="trust-note">
            <ShieldIcon />
            <span>Evidence-first answers with source citations</span>
          </div>
        </section>

        <section className="chat-card" aria-label="NovaCart assistant chat">
          <div className="chat-card__header">
            <div>
              <span className="chat-card__signal" aria-hidden="true" />
              Knowledge workspace
            </div>
            <span>{conversationId ? 'Conversation active' : 'New conversation'}</span>
          </div>

          <div ref={messagesRef} className="messages" aria-live="polite">
            {messages.map((message) => (
              <Message key={message.id} message={message} theme={theme} />
            ))}
            {isLoading && <LoadingMessage theme={theme} />}
          </div>

          <div className="composer-area">
            {messages.length === 1 && (
              <div className="suggestions" aria-label="Suggested questions">
                {STARTER_QUESTIONS.map((question) => (
                  <button
                    type="button"
                    key={question}
                    onClick={() => submitMessage(question)}
                    disabled={isLoading}
                  >
                    {question}
                  </button>
                ))}
              </div>
            )}

            {error && (
              <div className="error-banner" role="alert">
                <span>{error}</span>
                <button type="button" onClick={() => setError('')}>
                  Dismiss
                </button>
              </div>
            )}

            <BorderBeam
              className="composer-beam"
              size="md"
              colorVariant="ocean"
              strength={theme === 'dark' ? 0.72 : 0.54}
              active
              duration={isLoading ? 2.1 : 3.5}
              theme={theme}
            >
              <form
                className="composer"
                onSubmit={(event) => {
                  event.preventDefault()
                  submitMessage()
                }}
              >
                <textarea
                  ref={textareaRef}
                  value={query}
                  onChange={(event) => {
                    setQuery(event.target.value)
                    event.target.style.height = 'auto'
                    event.target.style.height =
                      String(Math.min(event.target.scrollHeight, 132)) + 'px'
                  }}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask NovaCart anything..."
                  rows="1"
                  disabled={isLoading}
                  aria-label="Message NovaCart AI Assistant"
                />
                <MetalFx
                  className="send-metal"
                  preset="chromatic"
                  variant="circle"
                  innerShadow
                  strength={0.82}
                  glowGain={0.7}
                  paused={isLoading || !query.trim()}
                  normalizeHostStyles={false}
                  theme={theme}
                >
                  <button
                    type="submit"
                    className="send-button"
                    disabled={isLoading || !query.trim()}
                    aria-label="Send message"
                  >
                    <ArrowUpIcon />
                  </button>
                </MetalFx>
              </form>
            </BorderBeam>

            <div className="composer-footer">
              <span>Enter to send</span>
              <span aria-hidden="true">/</span>
              <span>Shift + Enter for a new line</span>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

function Message({ message, theme }) {
  const isAssistant = message.role === 'assistant'
  const showDebug = import.meta.env.VITE_SHOW_DEBUG === 'true'

  return (
    <article className={'message message--' + message.role}>
      {isAssistant && <AssistantAvatar theme={theme} />}
      <div className="message-content">
        <div className="message-label">
          {isAssistant ? 'NovaCart AI' : 'You'}
        </div>
        <div className="message-bubble">
          <p>{message.content}</p>
          {isAssistant && message.sources?.length > 0 && (
            <SourceList sources={message.sources} />
          )}
          {showDebug && message.route && (
            <div className="debug-line">
              Route: {message.route} / Retrieved:{' '}
              {message.metadata?.retrieval_count ?? 0} /{' '}
              {message.metadata?.latency_ms ?? 0} ms
            </div>
          )}
        </div>
      </div>
    </article>
  )
}

function AssistantAvatar({ theme, busy = false }) {
  return (
    <div className="message-avatar">
      <BotAvatar
        type="droid"
        face="eyes"
        state={busy ? 'working' : 'default'}
        size={40}
        speed={0.82}
        seed={0.62}
        shading="plastic"
        interactive={false}
        paused={!busy}
        theme={theme}
        aria-label={busy ? 'NovaCart AI working' : 'NovaCart AI'}
      />
    </div>
  )
}

function SourceList({ sources }) {
  return (
    <div className="source-section">
      <div className="source-heading">
        <LinkIcon />
        <span>Sources</span>
      </div>
      <div className="source-grid">
        {sources.map((source, index) => (
          <article
            className="source-card"
            key={
              source.evidence_id ||
              [source.document_name, source.section, index].join('-')
            }
          >
            <div className="source-icon" aria-hidden="true">
              {source.document_type?.toUpperCase() || 'DOC'}
            </div>
            <div className="source-card__content">
              <strong>{source.document_name}</strong>
              {source.section && <span>{source.section}</span>}
              <small>
                {source.page ? 'Page ' + source.page + ' / ' : ''}
                {source.document_type?.toUpperCase() || 'DOCUMENT'}
              </small>
            </div>
          </article>
        ))}
      </div>
    </div>
  )
}

function LoadingMessage({ theme }) {
  return (
    <article className="message message--assistant" aria-label="Loading answer">
      <AssistantAvatar theme={theme} busy />
      <div className="message-content">
        <div className="message-label">NovaCart AI</div>
        <div className="message-bubble loading-bubble" role="status">
          <div className="thinking-orb">
            <ThinkingOrb
              state="searching"
              size={64}
              speed={0.92}
              theme={theme}
              aria-label="Searching NovaCart sources"
            />
          </div>
          <div className="loading-copy">
            <strong>Searching NovaCart</strong>
            <span>Retrieving and checking the strongest evidence...</span>
          </div>
        </div>
      </div>
    </article>
  )
}

function ArrowUpIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 19V5M12 5 6.5 10.5M12 5l5.5 5.5" />
    </svg>
  )
}

function LinkIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M10.5 13.5a4 4 0 0 0 5.66 0l2.34-2.34a4 4 0 0 0-5.66-5.66l-1.34 1.34M13.5 10.5a4 4 0 0 0-5.66 0L5.5 12.84a4 4 0 0 0 5.66 5.66l1.34-1.34" />
    </svg>
  )
}

function ShieldIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3 5 6v5c0 4.55 2.9 8.58 7 10 4.1-1.42 7-5.45 7-10V6l-7-3Z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  )
}

export default App
