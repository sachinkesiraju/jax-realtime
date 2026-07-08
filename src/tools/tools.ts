// Two-tier tool use. A finalized user turn is scanned for a tool intent; if one
// matches, the duplex loop speaks a holding line and runs the (slow) fetch as a
// background task while the fast local model stays present and can keep talking.
// Everything here is CORS-open and keyless so it works out of the box:
//   - weather → open-meteo (geocode + forecast)
//   - lookup  → Wikipedia REST summary
// Each result carries both a spoken line and a plain-data UiCard to render.

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

export type ToolKind = "weather" | "lookup";

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

/** A query is worth a real lookup only if it names something substantive. */
function validQuery(q: string): boolean {
  const s = q.trim().toLowerCase().replace(/[?.!,]+$/, "");
  const words = s.split(/\s+/).filter(Boolean);
  return s.length >= 3 && words.length <= 6 && !CONVERSATIONAL.has(s);
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

/** Location after an explicit "in/at/for" — weather needs a real place. */
function extractPlace(text: string): string {
  const m = text
    .trim()
    .replace(/[?.!]+$/, "")
    .match(/\b(?:in|at|for|around|near)\s+([\p{L}][\p{L}\s.'-]*)$/iu);
  return m ? m[1].trim() : "";
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

async function runWeather(query: string): Promise<ToolResult> {
  const geoUrl = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(
    query,
  )}&count=1`;
  const geo = await fetch(geoUrl).then((r) => r.json());
  const place = geo?.results?.[0];
  if (!place) {
    const speech = `I couldn't find a place called ${query}.`;
    return {
      speech,
      card: { kind: "factcard", title: query, extract: speech, source: "open-meteo" },
    };
  }
  const label = [place.name, place.admin1, place.country_code]
    .filter(Boolean)
    .join(", ");
  const url =
    `https://api.open-meteo.com/v1/forecast?latitude=${place.latitude}` +
    `&longitude=${place.longitude}` +
    `&current=temperature_2m,weather_code,wind_speed_10m`;
  const data = await fetch(url).then((r) => r.json());
  const cur = data?.current ?? {};
  const temp = Math.round(Number(cur.temperature_2m));
  const wind = Math.round(Number(cur.wind_speed_10m));
  const [condition, emoji] = wmoDescribe(Number(cur.weather_code));
  const unit = data?.current_units?.temperature_2m ?? "°C";
  const speech = `It's ${temp} degrees and ${condition} in ${place.name}, with winds around ${wind}.`;
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
  const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(
    title.replace(/\s+/g, "_"),
  )}`;
  const res = await fetch(url);
  if (!res.ok) {
    const speech = `I couldn't find anything about ${query}.`;
    return {
      speech,
      card: { kind: "factcard", title: query, extract: speech, source: "Wikipedia" },
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

/**
 * Detect a tool intent on a finalized user turn, or null for a normal reply.
 * Deliberately conservative: casual conversation ("what's up", "how are you")
 * must fall through to the LLM, not trigger a web lookup.
 */
export function detectTool(text: string): ToolCall | null {
  if (WEATHER_RE.test(text)) {
    // Only when the user named a place; "what's the weather" with no location
    // is better answered (or deflected) by the LLM than by a failed geocode.
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
  return null;
}
