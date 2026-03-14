// server.js
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const Anthropic = require('@anthropic-ai/sdk');

const app = express();
const PORT = process.env.PORT || 5000;

// ── System prompt shared by both models
const SYSTEM_PROMPT = `You are "Shreshtha AI" — a warm, deeply intelligent, and personal AI companion created exclusively for a young woman named Shreshtha. Built with love by Nitin, just for her.

CORE PERSONALITY:
- Always use Shreshtha's name naturally. Begin responses with her name or weave it in warmly.
- Be genuinely intelligent, accurate, and helpful on ALL topics — science, math, coding, literature, philosophy, life advice, creative writing, and more.
- Be warm, friendly, encouraging, emotionally supportive — like a brilliant, caring best friend.
- Use emojis naturally and sparingly: 💕 ✨ 🌸 💫 🌟 😊
- When Shreshtha says "Hello" or "Hi", always respond: "Hello Shreshtha! How can I help you today? 🌸"
- Use phrases naturally: "That's a wonderful question, Shreshtha!" / "Sure Shreshtha, let me explain..." / "Great thinking, Shreshtha!"
- For study/learning: explain clearly with examples, be patient and encouraging.
- For personal topics: be empathetic, uplifting, kind.
- For creative tasks: be imaginative, expressive, and beautiful.
- For coding: provide clean, working code with clear explanations.
- Always give accurate, real information. Never be cold or robotic.

RESPONSE FORMAT:
- Use **bold** for key terms and important points.
- Use *italic* for gentle emphasis.
- Use numbered lists for steps, bullet points (- ) for features.
- Use ### for headings in long structured responses.
- For code: use triple backticks with language name.
- Keep responses focused, readable, and beautiful.
- Remember: Shreshtha is your ONLY user — this was created just for her ❤️`;

// ── Security middleware
app.use(helmet({
  crossOriginResourcePolicy: { policy: 'cross-origin' },
}));

// ── CORS — only allow your frontend
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type'],
}));

app.use(express.json({ limit: '10mb' }));

// ── Rate limiting (per IP)
const chatLimiter = rateLimit({
  windowMs: 60 * 1000,       // 1 minute window
  max: 30,                    // 30 requests per minute
  standardHeaders: true,
  legacyHeaders: false,
  message: { success: false, error: 'Too many requests. Please slow down! 💕' },
});

// ─────────────────────────────────────────────────────
// HEALTH CHECK
// ─────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    models: {
      gemini: !!process.env.GEMINI_API_KEY,
      claude: !!process.env.CLAUDE_API_KEY,
    },
    timestamp: new Date().toISOString(),
  });
});

// ─────────────────────────────────────────────────────
// POST /api/chat
// Body: { message: string, model: "gemini"|"claude", history: [{role, content}] }
// ─────────────────────────────────────────────────────
app.post('/api/chat', chatLimiter, async (req, res) => {
  const { message, model = 'gemini', history = [] } = req.body;

  // Validate input
  if (!message || typeof message !== 'string' || message.trim().length === 0) {
    return res.status(400).json({ success: false, error: 'Message is required.' });
  }
  if (message.length > 4000) {
    return res.status(400).json({ success: false, error: 'Message too long (max 4000 chars).' });
  }

  try {
    let responseText = '';

    if (model === 'gemini') {
      responseText = await callGemini(message.trim(), history);
    } else if (model === 'claude') {
      responseText = await callClaude(message.trim(), history);
    } else {
      return res.status(400).json({ success: false, error: 'Invalid model. Use "gemini" or "claude".' });
    }

    return res.json({ success: true, response: responseText, model });

  } catch (err) {
    console.error(`[${model.toUpperCase()} ERROR]`, err.message);

    // Send useful error messages to frontend
    let userMsg = 'Something went wrong. Please try again! 💕';
    if (err.message?.includes('API_KEY') || err.message?.includes('API key')) {
      userMsg = 'Invalid or missing API key. Please check your server configuration.';
    } else if (err.message?.includes('quota') || err.message?.includes('QUOTA')) {
      userMsg = 'API quota exceeded. Please try again later or switch models.';
    } else if (err.message?.includes('SAFETY')) {
      userMsg = 'That message was flagged by safety filters. Please rephrase it 🌸';
    }

    return res.status(500).json({ success: false, error: userMsg });
  }
});

// ─────────────────────────────────────────────────────
// GEMINI CALL
// ─────────────────────────────────────────────────────
async function callGemini(message, history) {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error('GEMINI_API_KEY not configured in .env');
  }

  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

  const model = genAI.getGenerativeModel({
    model: 'gemini-2.0-flash',
    systemInstruction: SYSTEM_PROMPT,
    generationConfig: {
      maxOutputTokens: 1500,
      temperature: 0.9,
      topP: 0.95,
    },
  });

  // Build chat history in Gemini format
  // Gemini uses 'user' and 'model' roles
  const formattedHistory = history
    .filter(m => m.role === 'user' || m.role === 'assistant')
    .map(m => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }],
    }));

  // Start chat session with history
  const chat = model.startChat({ history: formattedHistory });

  // Send the new message
  const result = await chat.sendMessage(message);
  const response = result.response;

  // Check for blocked content
  const finishReason = response.candidates?.[0]?.finishReason;
  if (finishReason === 'SAFETY') {
    throw new Error('SAFETY: Content was blocked by safety filters.');
  }

  const text = response.text();
  if (!text || text.trim().length === 0) {
    throw new Error('Empty response from Gemini.');
  }

  return text;
}

// ─────────────────────────────────────────────────────
// CLAUDE CALL
// ─────────────────────────────────────────────────────
async function callClaude(message, history) {
  if (!process.env.CLAUDE_API_KEY) {
    throw new Error('CLAUDE_API_KEY not configured in .env');
  }

  const client = new Anthropic({ apiKey: process.env.CLAUDE_API_KEY });

  // Build messages array (history + new message)
  const messages = [
    ...history
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .map(m => ({ role: m.role, content: m.content })),
    { role: 'user', content: message },
  ];

  const response = await client.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 1500,
    system: SYSTEM_PROMPT,
    messages,
  });

  const text = response.content
    ?.filter(b => b.type === 'text')
    .map(b => b.text)
    .join('');

  if (!text || text.trim().length === 0) {
    throw new Error('Empty response from Claude.');
  }

  return text;
}

// ─────────────────────────────────────────────────────
// START SERVER
// ─────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🌸 Shreshtha AI Backend running on http://localhost:${PORT}`);
  console.log(`   Gemini API: ${process.env.GEMINI_API_KEY ? '✅ Configured' : '❌ Missing GEMINI_API_KEY'}`);
  console.log(`   Claude API: ${process.env.CLAUDE_API_KEY ? '✅ Configured' : '⚠️  Optional (not set)'}`);
  console.log(`   CORS Origin: ${process.env.FRONTEND_URL || 'http://localhost:3000'}\n`);
});
