import { useEffect, useRef, useState } from 'react'
import './App.css'
import { ApiError, sendChatMessage } from './services/api'

const STARTER_QUESTIONS = [
  'What is the standard return window?',
  'What is the warranty of NCM-24?',
  'Which products have more than 12 months warranty?',
]

const WELCOME_MESSAGE = {
  id: 'welcome',
  role: 'assistant',
  content:
    'Ask me about NovaCart products, delivery, returns, warranties, or order policies. I will answer from the available company sources.',
  sources: [],
}

function App() {
  const [messages, setMessages] = useState([WELCOME_MESSAGE])
  const [query, setQuery] = useState('')
  const [conversationId, setConversationId] = useState(() =>
    sessionStorage.getItem('novacart_conversation_id'),
  )
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const messageEndRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' })
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
          <div className="brand-mark" aria-hidden="true">
            N
          </div>
          <div>
            <p className="eyebrow">NOVACART KNOWLEDGE</p>
            <h1>AI Assistant</h1>
          </div>
        </div>
        <div className="system-status">
          <span className="status-dot" aria-hidden="true" />
          Grounded answers
        </div>
      </header>

      <main className="chat-layout">
        <section className="intro-panel" aria-labelledby="intro-title">
          <p className="eyebrow">HYBRID RAG</p>
          <h2 id="intro-title">Answers backed by NovaCart sources.</h2>
          <p>
            Dense search, BM25, structured product data, and source citations
            work together behind one chat experience.
          </p>
          <div className="capability-list">
            <span>Products</span>
            <span>Policies</span>
            <span>Shipping</span>
            <span>Warranties</span>
          </div>
        </section>

        <section className="chat-card" aria-label="NovaCart assistant chat">
          <div className="messages" aria-live="polite">
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
            {isLoading && <LoadingMessage />}
            <div ref={messageEndRef} />
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
                  event.target.style.height = `${Math.min(
                    event.target.scrollHeight,
                    130,
                  )}px`
                }}
                onKeyDown={handleKeyDown}
                placeholder="Ask a NovaCart question…"
                rows="1"
                disabled={isLoading}
                aria-label="Message NovaCart AI Assistant"
              />
              <button
                type="submit"
                className="send-button"
                disabled={isLoading || !query.trim()}
                aria-label="Send message"
              >
                <SendIcon />
              </button>
            </form>
            <p className="composer-hint">
              Enter to send · Shift + Enter for a new line
            </p>
          </div>
        </section>
      </main>
    </div>
  )
}

function Message({ message }) {
  const isAssistant = message.role === 'assistant'
  const showDebug = import.meta.env.VITE_SHOW_DEBUG === 'true'

  return (
    <article className={`message message--${message.role}`}>
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
            Route: {message.route} · Retrieved:{' '}
            {message.metadata?.retrieval_count ?? 0} ·{' '}
            {message.metadata?.latency_ms ?? 0} ms
          </div>
        )}
      </div>
    </article>
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
            key={source.evidence_id || `${source.document_name}-${index}`}
          >
            <div className="source-icon" aria-hidden="true">
              {source.document_type?.toUpperCase() || 'DOC'}
            </div>
            <div>
              <strong>{source.document_name}</strong>
              {source.section && <span>{source.section}</span>}
              <small>
                {source.page ? `Page ${source.page} · ` : ''}
                {source.document_type?.toUpperCase() || 'DOCUMENT'}
              </small>
            </div>
          </article>
        ))}
      </div>
    </div>
  )
}

function LoadingMessage() {
  return (
    <article className="message message--assistant" aria-label="Loading answer">
      <div className="message-label">NovaCart AI</div>
      <div className="message-bubble loading-bubble">
        <span />
        <span />
        <span />
      </div>
    </article>
  )
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m4 4 17 8-17 8 3-7 8-1-8-1-3-7Z" />
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

export default App
