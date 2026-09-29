const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
const DEFAULT_TIMEOUT_MS = 150_000

export class ApiError extends Error {
  constructor(message, status = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export async function sendChatMessage({ query, conversationId = null }) {
  const controller = new AbortController()
  const configuredTimeout = Number(import.meta.env.VITE_API_TIMEOUT_MS)
  const timeoutMs =
    Number.isFinite(configuredTimeout) && configuredTimeout > 0
      ? configuredTimeout
      : DEFAULT_TIMEOUT_MS
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs)

  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        conversation_id: conversationId,
      }),
      signal: controller.signal,
    })

    const payload = await parsePayload(response)
    if (!response.ok) {
      throw new ApiError(
        payload?.detail || `NovaCart returned HTTP ${response.status}.`,
        response.status,
      )
    }
    if (!isChatResponse(payload)) {
      throw new ApiError('NovaCart returned an invalid chat response.')
    }
    return payload
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (error.name === 'AbortError') {
      throw new ApiError('The request timed out. Please try again.')
    }
    throw new ApiError(
      'The NovaCart API is unavailable. Check that the backend is running.',
    )
  } finally {
    window.clearTimeout(timeout)
  }
}

async function parsePayload(response) {
  try {
    return await response.json()
  } catch {
    return null
  }
}

function isChatResponse(payload) {
  return (
    payload &&
    typeof payload.conversation_id === 'string' &&
    typeof payload.answer === 'string' &&
    typeof payload.route === 'string' &&
    Array.isArray(payload.sources) &&
    payload.metadata &&
    typeof payload.metadata.retrieval_count === 'number'
  )
}
