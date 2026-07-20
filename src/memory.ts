const MEMORY_KINDS = [
  "name",
  "location",
  "trip",
  "pet",
  "favorite",
  "plan",
  "interview",
  "relation",
  "event",
  "activity",
] as const;

type MemoryKind = (typeof MEMORY_KINDS)[number];

export type MemoryFact = {
  kind: MemoryKind;
  value: string;
  turn: number;
};

export type ConversationalMemory = Partial<Record<MemoryKind, MemoryFact>>;

const CLAUSE_END = /\s+(?:and|but)\s+(?=(?:my|i\b|i'm|i’ve|i've)\s)/i;

function cleanValue(value: string): string {
  return value
    .split(CLAUSE_END)[0]
    .replace(/,?\s+by the way$/i, "")
    .replace(/^[\s,:;-]+|[\s,.!?;:-]+$/g, "")
    .replace(/\s+/g, " ")
    .replace(/[\[\]{}]/g, "")
    .trim();
}

function addFact(
  facts: MemoryFact[],
  kind: MemoryKind,
  value: string | undefined,
  turn: number,
): void {
  if (!value || facts.some((fact) => fact.kind === kind)) return;
  const cleaned = cleanValue(value);
  if (!cleaned || cleaned.length > 120) return;
  facts.push({ kind, value: cleaned, turn });
}

function extractExplicitUserFacts(text: string, turn: number): MemoryFact[] {
  const normalized = text.replace(/[’]/g, "'").replace(/\s+/g, " ").trim();
  const facts: MemoryFact[] = [];
  let match: RegExpMatchArray | null;

  match = normalized.match(
    /\b(?:my name is|call me|i am called|i'm called)\s+([^.!?]+)/iu,
  );
  addFact(facts, "name", match?.[1], turn);
  if (!facts.some((fact) => fact.kind === "name")) {
    match = normalized.match(/\b(?:I'm|I am)\s+([A-Z][\p{L}'-]{1,30})\b/u);
    const blocked = new Set([
      "From",
      "Going",
      "Traveling",
      "Travelling",
      "Flying",
      "Planning",
      "Interviewing",
      "Living",
      "Located",
    ]);
    if (match && !blocked.has(match[1])) addFact(facts, "name", match[1], turn);
  }

  match = normalized.match(
    /\b(?:i live in|i'm from|i am from|my home is in|my location is|i (?:just )?moved to)\s+([^.!?]+)/i,
  );
  addFact(facts, "location", match?.[1], turn);

  match = normalized.match(
    /\b(?:i'm|i am)\s+(?:traveling|travelling|flying)\s+to\s+([^.!?]+)|\b(?:i will|i'm going to|i am going to|i plan to)\s+(?:travel|fly|visit)\s+(?:to\s+)?([^.!?]+)|\bmy trip is\s+(?:to\s+)?([^.!?]+)|\bi(?:'m| am) planning a trip to\s+([^.!?]+)/i,
  );
  addFact(facts, "trip", match?.[1] ?? match?.[2] ?? match?.[3] ?? match?.[4], turn);

  match = normalized.match(
    /\bmy\s+(cat|dog|pet|rabbit|bird|hamster|horse)(?:'s name is| is named| is called)\s+([^.!?]+)|\bi have\s+(?:a|an)\s+(cat|dog|rabbit|bird|hamster|horse)\s+(?:named|called)\s+([^.!?]+)|\bi (?:adopted|got|have)\s+(?:a|an)\s+(puppy|kitten|cat|dog|rabbit|bird|hamster|horse)(?:\s+([^.!?]+))?/i,
  );
  if (match) {
    const animal = (match[1] ?? match[3] ?? match[5]).toLowerCase();
    const petName = match[2] ?? match[4];
    addFact(facts, "pet", petName ? `${animal} named ${petName}` : animal, turn);
  }

  match = normalized.match(
    /\bmy favou?rite\s+([\p{L}][\p{L} -]{0,30}?)\s+is\s+([^.!?]+)/iu,
  );
  if (match) addFact(facts, "favorite", `${match[1].trim()}: ${match[2]}`, turn);

  match = normalized.match(
    /\b(?:i have|i've got)\s+(?:an|a|my)\s+(?:[\p{L}-]+\s+)?interview\s*([^.!?]*)|\bmy interview is\s*([^.!?]+)|\b(?:i'm|i am)\s+interviewing\s*([^.!?]*)/iu,
  );
  if (match) {
    const detail = cleanValue(match[1] ?? match[2] ?? match[3] ?? "");
    addFact(facts, "interview", detail ? `interview ${detail}` : "interview", turn);
  }

  match = normalized.match(
    /\b(?:i plan to|i'm planning to|i am planning to|my plan is to|i intend to)\s+([^.!?]+)/i,
  );
  addFact(facts, "plan", match?.[1], turn);

  match = normalized.match(
    /\bmy\s+(sister|brother|mother|mom|father|dad|friend|partner|wife|husband|daughter|son|cousin|aunt|uncle|colleague)\s+([A-Z][\p{L}'-]{1,30})\s+([^.!?]+)/iu,
  );
  if (match) addFact(facts, "relation", `${match[1]} ${match[2]}`, turn);

  match = normalized.match(
    /\bi\s+(burned|burnt|broke|lost|missed|finished|started|made|cooked)\s+([^.!?]+)/i,
  );
  if (match) addFact(facts, "event", `${match[1]} ${match[2]}`, turn);

  match = normalized.match(
    /\bi(?:'ve| have) been\s+(learning|practicing|practising|playing|training)\s+([^.!?]+)/i,
  );
  if (match) addFact(facts, "activity", `${match[1]} ${match[2]}`, turn);

  return facts;
}

export function rememberUserFacts(
  memory: ConversationalMemory,
  text: string,
  turn: number,
): ConversationalMemory {
  const next = { ...memory };
  for (const fact of extractExplicitUserFacts(text, turn)) next[fact.kind] = fact;
  return next;
}

function recent(memory: ConversationalMemory, kind: MemoryKind, turn: number): boolean {
  const fact = memory[kind];
  return !!fact && turn - fact.turn <= 3;
}

export function relevantMemoryFacts(
  memory: ConversationalMemory,
  text: string,
  turn: number,
): MemoryFact[] {
  const s = text.toLowerCase().replace(/[’]/g, "'");
  const newlyStated = new Set(extractExplicitUserFacts(text, turn).map((fact) => fact.kind));
  const wanted = new Set<MemoryKind>();

  if (/\b(?:what(?:'s| is| was) my name|who am i|do you remember my name|what did i (?:say|tell you) my name was|what should you call me)\b/.test(s)) wanted.add("name");
  if (/\b(?:where do i live|where am i from|what(?:'s| is) my (?:city|location)|do you remember where i live|weather (?:where i live|near me))\b/.test(s)) wanted.add("location");
  if (/\b(?:my trip|my vacation|my journey|where am i (?:going|traveling|travelling|flying)|when am i (?:leaving|traveling|travelling|flying)|what should i pack|packing|itinerary|passport|flight|hotel)\b/.test(s)) wanted.add("trip");
  if (recent(memory, "trip", turn) && /\b(?:there|while i'm there|food|eat|try|visit|stay)\b/.test(s)) wanted.add("trip");
  if (/\b(?:my pet|my cat|my dog|do you remember (?:my|the) (?:pet|cat|dog)|what(?:'s| is) my (?:pet|cat|dog)(?:'s)? name)\b/.test(s)) wanted.add("pet");
  if (recent(memory, "pet", turn) && /\bwhat(?:'s| is) (?:his|her|its) name\b/.test(s)) wanted.add("pet");
  if (/\b(?:what(?:'s| is) my favou?rite|do you remember my favou?rite|what do i (?:like|love|prefer)|which .+ do i (?:like|love|prefer))\b/.test(s)) wanted.add("favorite");
  if (/\b(?:my plans?|what (?:am i|was i) planning|what did i (?:plan|say i was going) to do|do you remember my plans?)\b/.test(s)) wanted.add("plan");
  if (/\b(?:my interview|the interview|when is my interview|where is my interview|who am i interviewing with|interview prep|prepare for (?:it|the interview))\b/.test(s)) wanted.add("interview");
  if (recent(memory, "interview", turn) && /\b(?:wear|prepare|bring|say)\b|\bto it\b/.test(s)) wanted.add("interview");
  if (recent(memory, "pet", turn) && /\b(?:him|her|it)\b/.test(s) && /\b(?:feed|teach|train|give|buy|get|name)\b/.test(s)) wanted.add("pet");
  if (recent(memory, "favorite", turn) && /\b(?:what|which)\s+(?:color|colour)\b/.test(s)) wanted.add("favorite");
  if (recent(memory, "relation", turn) && /\b(?:we|us|her|him|together)\b/.test(s)) wanted.add("relation");
  if (recent(memory, "event", turn) && /\b(?:it|instead|now|what should i do|should i)\b/.test(s)) wanted.add("event");
  if (
    recent(memory, "activity", turn) &&
    /\b(?:finger|fingers|hand|hands|hurt|sore|pain|normal|practice|practicing|practising)\b/.test(s)
  ) {
    wanted.add("activity");
  }

  if (/\b(?:any advice|what should i do|how should i prepare|can you help me prepare|what do you think)\b/.test(s)) {
    if (recent(memory, "interview", turn)) wanted.add("interview");
    else if (recent(memory, "trip", turn)) wanted.add("trip");
    else if (recent(memory, "plan", turn)) wanted.add("plan");
    else if (recent(memory, "event", turn)) wanted.add("event");
  }

  return MEMORY_KINDS.flatMap((kind) => {
    const fact = memory[kind];
    return fact && wanted.has(kind) && !newlyStated.has(kind) ? [fact] : [];
  });
}

function factSentence(fact: MemoryFact): string {
  switch (fact.kind) {
    case "name":
      return `The user's name is ${fact.value}.`;
    case "location":
      return `The user lives in or is from ${fact.value}.`;
    case "trip":
      return `The user's trip is ${fact.value}.`;
    case "pet":
      return `The user has a ${fact.value}.`;
    case "favorite": {
      const [topic, ...value] = fact.value.split(":");
      return `The user's favorite ${topic} is ${value.join(":").trim()}.`;
    }
    case "plan":
      return `The user plans to ${fact.value}.`;
    case "interview":
      return `The user has an ${fact.value}.`;
    case "relation":
      return `The user mentioned their ${fact.value}.`;
    case "event":
      return `The user ${fact.value}.`;
    case "activity":
      return `The user has been ${fact.value}.`;
  }
}

export function directMemoryAnswer(
  facts: readonly MemoryFact[],
  text: string,
): string | null {
  const s = text.toLowerCase().replace(/[’]/g, "'");
  const fact = (kind: MemoryKind) => facts.find((candidate) => candidate.kind === kind);
  if (/\b(?:what(?:'s| is| was) my name|who am i|what should you call me)\b/.test(s)) {
    const name = fact("name");
    if (name) return `Your name is ${name.value}.`;
  }
  if (/\b(?:where do i live|where am i from|what(?:'s| is) my (?:city|location))\b/.test(s)) {
    const location = fact("location");
    if (location) return `You said you live in or are from ${location.value}.`;
  }
  if (/\b(?:what(?:'s| is) my favou?rite|do you remember my favou?rite)\b/.test(s)) {
    const favorite = fact("favorite");
    if (favorite) {
      const [topic, ...value] = favorite.value.split(":");
      return `Your favorite ${topic} is ${value.join(":").trim()}.`;
    }
  }
  if (/\b(?:what|which)\s+(?:color|colour)\b/.test(s)) {
    const favorite = fact("favorite");
    if (favorite?.value.toLowerCase().startsWith("color:")) {
      return `Since ${favorite.value.slice(6).trim()} is your favorite color, that would be a natural choice.`;
    }
  }
  if (/\bwhat(?:'s| is) my (?:pet|cat|dog)(?:'s)? name\b/.test(s)) {
    const pet = fact("pet");
    const named = pet?.value.match(/named\s+(.+)$/i)?.[1];
    if (named) return `Your pet's name is ${named}.`;
  }
  return null;
}

export function injectMemoryTag(content: string, facts: readonly MemoryFact[]): string {
  return facts.length
    ? `[memory: ${facts.map(factSentence).join(" ")}] ${content}`
    : content;
}
