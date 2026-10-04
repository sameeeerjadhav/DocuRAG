const INSUFFICIENT = "I don't have enough information";

export function plainAnswer(content) {
  return String(content || "")
    .replace(/[*_`#]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function isInsufficient(content) {
  const text = plainAnswer(content);
  return text === INSUFFICIENT || text.startsWith(`${INSUFFICIENT} `);
}
