// Two-tier tool use. A finalized user turn is scanned for a tool intent; if one
// matches, the duplex loop speaks a holding line and runs the (slow) fetch as a
// background task while the fast local model stays present and can keep talking.
// Everything here is CORS-open and keyless so it works out of the box:
//   - weather → open-meteo (geocode + forecast)
//   - lookup  → Wikipedia REST summary
//   - calc / clock → instant, fully offline (TUNABLES.toolRouting
//     "broad"; the GPT-Live-style posture of delegating what the tiny local
//     model can't answer reliably)
// Each result carries both a spoken line and a plain-data UiCard to render.

import { TUNABLES } from "../tunables";

export type UiCard =
  | {
      kind: "weather";
      location: string;
      temperature: number;
      condition: string;
      emoji: string;
      wind: number;
    }
  | { kind: "factcard"; title: string; extract: string; source: string }
  | { kind: "list"; title: string; items: string[] };

export type ToolResult = { speech: string; card: UiCard };

export type ToolKind = "weather" | "lookup" | "calc" | "clock";

export interface ToolCall {
  kind: ToolKind;
  query: string;
  /** Holding line spoken immediately while run() works in the background. */
  holding: string;
  run(): Promise<ToolResult>;
}

const WEATHER_RE = /\b(weather|forecast|temperature|how (?:hot|cold|warm))\b/i;
// Only EXPLICIT lookup intent — never bare "what's …", which fires on greetings
// like "what's up". "Who is <name>" is allowed; open questions ("what is the
// capital of France") are left to the LLM, which answers them fine.
const LOOKUP_RE =
  /\b(search for|look up|looking up|look for|google|tell me about|who (?:is|was|are|were))\b/i;

// Casual phrases that must never trigger a tool even if they brush a keyword.
const CONVERSATIONAL = new Set([
  "up",
  "it",
  "that",
  "this",
  "good",
  "new",
  "going on",
  "happening",
  "you",
  "your name",
  "the time",
  "the date",
  "the deal",
  "cracking",
  "poppin",
  "poppin'",
  "the matter",
  "wrong",
]);

// A pronoun as the subject means the query is conversational or refers to
// context ("tell me about them", "what are you saying") — never a Wikipedia
// entity. Falls through to the LLM, which has the conversation + scene context.
const PRONOUNS = new Set([
  "i", "me", "my", "mine", "myself",
  "you", "your", "yours", "yourself",
  "we", "us", "our", "ours",
  "they", "them", "their", "theirs",
  "he", "him", "his", "she", "her", "hers",
  "it", "its", "this", "that", "these", "those",
]);

// Deictic time adverbs and place prepositions — closed grammatical classes (like
// PRONOUNS), stripped from the LEADING edge of an extracted place phrase so
// "forecast for tomorrow" yields no place (→ the LLM answers) and "tomorrow in
// Paris" → "Paris". Trailing time words are left to geocodePlace's token-drop
// backoff, so only the leading edge is handled here.
const TIME_DEICTICS = new Set(["today", "tomorrow", "tonight", "yesterday", "now"]);
const PLACE_PREPS = new Set(["in", "at", "for", "around", "near", "on", "by"]);

/** A query is worth a real lookup only if it names something substantive. */
function validQuery(q: string): boolean {
  const s = q.trim().toLowerCase().replace(/[?.!,]+$/, "");
  const words = s.split(/\s+/).filter(Boolean);
  if (s.length < 3 || words.length === 0 || words.length > 6) return false;
  if (CONVERSATIONAL.has(s)) return false;
  // Reject pronoun-led queries: "them", "you saying", "that thing".
  if (PRONOUNS.has(words[0])) return false;
  return true;
}

/**
 * Pull a trailing location/entity phrase out of a query — the words after an
 * "in/for/at/about" preposition, or the tail after the trigger phrase.
 */
function extractTail(text: string, trigger: RegExp): string {
  const cleaned = text.trim().replace(/[?.!]+$/, "");
  const prep = cleaned.match(
    /\b(?:in|for|at|about|of|on)\s+([\p{L}][\p{L}\s.'-]*)$/iu,
  );
  if (prep) return prep[1].trim();
  const afterTrigger = cleaned.replace(trigger, "").trim();
  // Drop leading filler words.
  return afterTrigger.replace(/^(the|a|an|me|is|are|was|about)\s+/i, "").trim();
}

/**
 * The phrase after an "in/at/near" preposition — the place, plus whatever
 * conversational tail followed it ("San Francisco instead", "Paris right now").
 * We deliberately DON'T try to strip the tail with a word list; the geocoder is
 * the authority on what's a real place, so `geocodePlace` resolves the longest
 * leading run it recognizes. We only cut at clause punctuation (a structural
 * boundary, not a keyword) to bound the search.
 */
function extractPlace(text: string): string {
  const m = text.match(/\b(?:in|at|for|around|near)\s+([\p{L}][\p{L}\s.'-]*)/iu);
  if (!m) return "";
  const phrase = m[1].replace(/[.?!,;].*$/s, "").trim();
  // Strip leading deictic time words / prepositions (closed classes) so a bare
  // "for tomorrow" resolves to "" (→ tool doesn't fire) and "tomorrow in Paris"
  // → "Paris". The geocoder still owns the trailing tail (see geocodePlace).
  const words = phrase.split(/\s+/).filter(Boolean);
  while (
    words.length &&
    (TIME_DEICTICS.has(words[0].toLowerCase()) ||
      PLACE_PREPS.has(words[0].toLowerCase()))
  ) {
    words.shift();
  }
  return words.join(" ");
}

// Compact WMO weather-code → (text, emoji) map for the current-conditions card.
const WMO: Record<number, [string, string]> = {
  0: ["clear sky", "☀️"],
  1: ["mainly clear", "🌤️"],
  2: ["partly cloudy", "⛅"],
  3: ["overcast", "☁️"],
  45: ["foggy", "🌫️"],
  48: ["rime fog", "🌫️"],
  51: ["light drizzle", "🌦️"],
  53: ["drizzle", "🌦️"],
  55: ["heavy drizzle", "🌧️"],
  61: ["light rain", "🌦️"],
  63: ["rain", "🌧️"],
  65: ["heavy rain", "🌧️"],
  71: ["light snow", "🌨️"],
  73: ["snow", "🌨️"],
  75: ["heavy snow", "❄️"],
  80: ["rain showers", "🌦️"],
  81: ["rain showers", "🌧️"],
  82: ["violent rain showers", "⛈️"],
  95: ["thunderstorm", "⛈️"],
  96: ["thunderstorm with hail", "⛈️"],
  99: ["thunderstorm with hail", "⛈️"],
};

function wmoDescribe(code: number): [string, string] {
  return WMO[code] ?? ["unknown conditions", "🌡️"];
}

// Resolve a place by asking the geocoder — the authority on place names —
// rather than hand-cleaning the query. Try the whole phrase, then drop trailing
// tokens until it matches, so "San Francisco instead" / "Paris right now" find
// "San Francisco" / "Paris" without any junk-word list.
async function geocodePlace(
  query: string,
): Promise<{ name: string; admin1?: string; country_code?: string; latitude: number; longitude: number } | null> {
  const tokens = query.split(/\s+/).filter(Boolean);
  for (let end = tokens.length; end >= 1; end--) {
    const name = tokens.slice(0, end).join(" ");
    const geo = await fetch(
      `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(name)}&count=1`,
    ).then((r) => r.json());
    if (geo?.results?.[0]) return geo.results[0];
  }
  return null;
}

async function runWeather(query: string): Promise<ToolResult> {
  const place = await geocodePlace(query);
  if (!place) {
    // Never SPEAK the raw query — ASR often garbles a place name, and reading it
    // back is worse than asking. The card keeps it so the user sees what we heard.
    const speech = "Hmm, I couldn't find that place — which city did you mean?";
    return {
      speech,
      card: {
        kind: "factcard",
        title: query,
        extract: `I couldn't find a place called ${query}.`,
        source: "open-meteo",
      },
    };
  }
  const label = [place.name, place.admin1, place.country_code]
    .filter(Boolean)
    .join(", ");
  const url =
    `https://api.open-meteo.com/v1/forecast?latitude=${place.latitude}` +
    `&longitude=${place.longitude}` +
    `&current=temperature_2m,weather_code,wind_speed_10m` +
    // Fahrenheit + mph by default (US-oriented). open-meteo does the unit
    // conversion server-side, so temp/wind come back already in these units.
    `&temperature_unit=fahrenheit&wind_speed_unit=mph`;
  const data = await fetch(url).then((r) => r.json());
  const cur = data?.current ?? {};
  const temp = Math.round(Number(cur.temperature_2m));
  const wind = Math.round(Number(cur.wind_speed_10m));
  const [condition, emoji] = wmoDescribe(Number(cur.weather_code));
  const unit = data?.current_units?.temperature_2m ?? "°F";
  const speech = `It's ${temp} degrees and ${condition} in ${place.name}, with winds around ${wind} miles per hour.`;
  return {
    speech,
    card: {
      kind: "weather",
      location: label,
      temperature: temp,
      condition: `${condition} · ${temp}${unit}`,
      emoji,
      wind,
    },
  };
}

async function runLookup(query: string): Promise<ToolResult> {
  const title = query
    .split(/\s+/)
    .map((w) => (w.length > 2 ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
  // NOTE (campaign 2): a title-search fallback for phrase queries was tried and
  // rejected — it turned honest misses into confidently irrelevant answers
  // ("how far is the moon" → a 2007 film). An honest "couldn't find" is the
  // right behavior until a snippet-based retrieval source exists.
  const res = await fetch(
    `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(
      title.replace(/\s+/g, "_"),
    )}`,
  );
  if (!res.ok) {
    // Query-free spoken line (the raw query is often ASR-garbled); the card still
    // shows it so the user can see what was heard and rephrase.
    const speech = "I couldn't find that one — could you say it a different way?";
    return {
      speech,
      card: {
        kind: "factcard",
        title: query,
        extract: `I couldn't find anything about ${query}.`,
        source: "Wikipedia",
      },
    };
  }
  const data = await res.json();
  const extract: string = data.extract ?? "No summary available.";
  const firstSentence = extract.split(/(?<=[.!?])\s/)[0] ?? extract;
  return {
    speech: firstSentence.slice(0, 240),
    card: {
      kind: "factcard",
      title: data.title ?? title,
      extract,
      source: "Wikipedia",
    },
  };
}

// --- Instant deterministic tools (broad routing only) -----------------------

const NUM = "(-?\\d+(?:[.,]\\d+)?)";
const num = (s: string) => Number(s.replace(",", "."));
const fmt = (x: number) =>
  Math.abs(x - Math.round(x)) < 1e-9 ? String(Math.round(x)) : x.toFixed(2);

/** Spoken-arithmetic parser: "17 times 23", "15 percent of 80", "144 / 12". */
function detectCalc(text: string): ToolCall | null {
  const t = text.toLowerCase().replace(/[?.!]+$/, "");
  // Word operators bind loosely (`\s*`); bare SYMBOL operators (- + / * x) must be
  // whitespace-separated on BOTH sides (`\s+`) so ranges and dimensions parse as
  // text, not arithmetic: "3 - 4" is subtraction, but "3-4 people", "3/4",
  // "1920x1080" are not. (A leading "-" in NUM still lets "3 - -4" mean 3 − (−4).)
  const patterns: [RegExp, (a: number, b: number) => number, string][] = [
    [new RegExp(`${NUM}(?:\\s*(?:times|multiplied by)\\s*|\\s+(?:x|\\*)\\s+)${NUM}`), (a, b) => a * b, "times"],
    [new RegExp(`${NUM}(?:\\s*(?:divided by|over)\\s*|\\s+/\\s+)${NUM}`), (a, b) => a / b, "divided by"],
    [new RegExp(`${NUM}(?:\\s*plus\\s*|\\s+\\+\\s+)${NUM}`), (a, b) => a + b, "plus"],
    [new RegExp(`${NUM}(?:\\s*minus\\s*|\\s+-\\s+)${NUM}`), (a, b) => a - b, "minus"],
    [new RegExp(`${NUM}\\s*(?:percent|%)\\s*of\\s*${NUM}`), (a, b) => (a / 100) * b, "percent of"],
  ];
  for (const [re, op, word] of patterns) {
    const m = t.match(re);
    if (!m) continue;
    const a = num(m[1]);
    const b = num(m[2]);
    const result = op(a, b);
    if (!Number.isFinite(result)) continue;
    const speech = `${fmt(a)} ${word} ${fmt(b)} is ${fmt(result)}.`;
    return {
      kind: "calc",
      query: m[0],
      holding: "",
      run: async () => ({
        speech,
        card: { kind: "factcard", title: `${fmt(result)}`, extract: speech, source: "calculator" },
      }),
    };
  }
  return null;
}

// Unit conversions: factor to the target unit (linear), plus temperature.
// NOTE (owner call): a unit-conversion tool lived here briefly and was deleted.
// Conversion factors are stored knowledge — a hand-maintained mini-almanac with
// no principled stopping point (currencies? time zones?) — which is the
// hardcoding pattern this project rejects. Calculator (pure arithmetic) and
// clock (device state) are capabilities, not knowledge, so they stay.

const CLOCK_RE =
  /\b(what day is (it|today)|what('s| is) (the date|today's date)|what time is it|what('s| is) the time)\b/i;

/** Local date/time — the one thing even a big LLM can't know offline. */
function detectClock(text: string): ToolCall | null {
  if (!CLOCK_RE.test(text)) return null;
  const now = new Date();
  const wantsTime = /time/i.test(text);
  const speech = wantsTime
    ? `It's ${now.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}.`
    : `It's ${now.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric", year: "numeric" })}.`;
  return {
    kind: "clock",
    query: wantsTime ? "time" : "date",
    holding: "",
    run: async () => ({
      speech,
      card: { kind: "factcard", title: speech.replace(/^It's /, ""), extract: speech, source: "clock" },
    }),
  };
}

// Broad lookup routing: wh-questions route to Wikipedia (which answers factual
// questions far more reliably than a 270M model). Small-talk stays protected by
// validQuery + the stoplist, exactly as in conservative mode.
const BROAD_LOOKUP_RE = /\b(?:what|who|where)(?:'s| is| are| was| were)\b/i;
const MEANING_RE = /\bwhat does\s+(.+?)\s+mean\b/i;
const HOWFAR_RE = /\bhow (?:far|big|tall|old|heavy|deep|long) is\b/i;

function broadLookupQuery(text: string): string {
  const meaning = text.match(MEANING_RE);
  if (meaning) return meaning[1].trim();
  const cleaned = text.trim().replace(/[?.!]+$/, "");
  // Strip the leading wh-phrase and articles: "What is the Eiffel Tower" →
  // "Eiffel Tower"; "How far is the moon from Earth" → "moon from Earth".
  return cleaned
    .replace(/^(?:what|who|where|how (?:far|big|tall|old|heavy|deep|long))(?:'s| is| are| was| were)?\s+/i, "")
    .replace(/^(the|a|an)\s+/i, "")
    .trim();
}

/**
 * Detect a tool intent on a finalized user turn, or null for a normal reply.
 * Deliberately conservative: casual conversation ("what's up", "how are you")
 * must fall through to the LLM, not trigger a web lookup.
 */
export function detectTool(text: string): ToolCall | null {
  const broad = TUNABLES.toolRouting === "broad";

  // Instant deterministic tools first — their patterns are precise (numbers,
  // explicit clock phrases) so they can't shadow conversation.
  if (broad) {
    const instant = detectCalc(text) ?? detectClock(text);
    if (instant) return instant;
  }

  if (WEATHER_RE.test(text)) {
    // Only when the user named a place; "what's the weather" with no location
    // is better answered (or deflected) by the LLM than by a failed geocode.
    // The place may carry a conversational tail ("San Francisco instead") —
    // geocodePlace resolves it, so no query cleanup here.
    const query = extractPlace(text);
    if (validQuery(query)) {
      return {
        kind: "weather",
        query,
        holding: "Let me check the weather.",
        run: () => runWeather(query),
      };
    }
    return null;
  }

  if (LOOKUP_RE.test(text)) {
    const query = extractTail(text, LOOKUP_RE);
    if (validQuery(query)) {
      return {
        kind: "lookup",
        query,
        holding: "Let me look that up.",
        run: () => runLookup(query),
      };
    }
  }

  // Broad mode: wh-questions become Wikipedia lookups — factual questions are
  // the 270M model's weakest axis, and delegation answers them reliably. The
  // stoplist/validQuery guard keeps small talk ("what's up") with the LLM.
  if (broad && (BROAD_LOOKUP_RE.test(text) || HOWFAR_RE.test(text) || MEANING_RE.test(text))) {
    const query = broadLookupQuery(text);
    if (validQuery(query)) {
      return {
        kind: "lookup",
        query,
        holding: "Let me look that up.",
        run: () => runLookup(query),
      };
    }
  }

  return null;
}
