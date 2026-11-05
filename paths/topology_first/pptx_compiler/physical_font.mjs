import { spawnSync } from "node:child_process";

const cache = new Map();
export function physicalFont(requested = "Noto Sans") {
  if (cache.has(requested)) return cache.get(requested);
  const exact = (family) => ["Regular", "Bold"].every((style) => {
    const result = spawnSync("fc-match", ["-f", "%{family}\n%{style}", `${family}:style=${style}`], { encoding: "utf8" });
    const [families = "", weight = ""] = String(result.stdout || "").trim().split("\n");
    return result.status === 0 && families.split(",").map((s) => s.trim()).includes(family)
      && (style !== "Bold" || weight.toLowerCase().includes("bold"));
  });
  const resolved = exact(requested) ? requested : "Noto Sans";
  if (!exact(resolved)) throw new Error(`Required physical deck font unavailable: ${resolved}`);
  cache.set(requested, resolved);
  return resolved;
}
