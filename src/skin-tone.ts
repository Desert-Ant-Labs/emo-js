/** Preferred emoji skin tone variant for skin-tone-capable emoji. */
export type EmojiSkinTone = "default" | "light" | "mediumLight" | "medium" | "mediumDark" | "dark";

/** Options for emoji suggestion post-processing. */
export interface EmoSuggestionOptions {
  /** Preferred skin tone for skin-tone-capable emoji. Defaults to `"default"` (no modifier). */
  skinTone?: EmojiSkinTone;
}

const modifiers: Record<Exclude<EmojiSkinTone, "default">, string> = {
  light: "🏻",
  mediumLight: "🏼",
  medium: "🏽",
  mediumDark: "🏾",
  dark: "🏿",
};

const isModifier = (cp: number): boolean => cp >= 0x1f3fb && cp <= 0x1f3ff;
const isVariationSelector16 = (cp: number): boolean => cp === 0xfe0f;

function isEmojiModifierBase(cp: number): boolean {
  return cp === 0x261d || cp === 0x26f9 ||
    (cp >= 0x270a && cp <= 0x270d) || cp === 0x1f385 ||
    (cp >= 0x1f3c2 && cp <= 0x1f3c4) || cp === 0x1f3c7 ||
    (cp >= 0x1f3ca && cp <= 0x1f3cc) || (cp >= 0x1f442 && cp <= 0x1f443) ||
    (cp >= 0x1f446 && cp <= 0x1f450) || (cp >= 0x1f466 && cp <= 0x1f478) ||
    cp === 0x1f47c || (cp >= 0x1f481 && cp <= 0x1f483) ||
    (cp >= 0x1f485 && cp <= 0x1f487) || cp === 0x1f48f || cp === 0x1f491 ||
    cp === 0x1f4aa || (cp >= 0x1f574 && cp <= 0x1f575) || cp === 0x1f57a ||
    cp === 0x1f590 || (cp >= 0x1f595 && cp <= 0x1f596) ||
    (cp >= 0x1f645 && cp <= 0x1f647) || (cp >= 0x1f64b && cp <= 0x1f64f) ||
    cp === 0x1f6a3 || (cp >= 0x1f6b4 && cp <= 0x1f6b6) || cp === 0x1f6c0 ||
    cp === 0x1f6cc || cp === 0x1f90c || (cp >= 0x1f918 && cp <= 0x1f91f) ||
    cp === 0x1f926 || (cp >= 0x1f930 && cp <= 0x1f939) ||
    (cp >= 0x1f93d && cp <= 0x1f93e) || cp === 0x1f977 ||
    (cp >= 0x1f9b5 && cp <= 0x1f9b6) || (cp >= 0x1f9b8 && cp <= 0x1f9b9) ||
    (cp >= 0x1f9cd && cp <= 0x1f9cf) || (cp >= 0x1f9d1 && cp <= 0x1f9dd) ||
    (cp >= 0x1faf0 && cp <= 0x1faf8);
}

export function applyEmojiSkinTone(emoji: string, skinTone: EmojiSkinTone = "default"): string {
  if (skinTone === "default") return emoji;
  const modifier = modifiers[skinTone];
  const chars = Array.from(emoji);
  let out = "";
  for (let i = 0; i < chars.length; i++) {
    const cp = chars[i].codePointAt(0)!;
    if (isModifier(cp)) continue;
    out += chars[i];
    if (isEmojiModifierBase(cp)) {
      out += modifier;
      while (i + 1 < chars.length) {
        const next = chars[i + 1].codePointAt(0)!;
        if (!isModifier(next) && !isVariationSelector16(next)) break;
        i++;
      }
    }
  }
  return out;
}
