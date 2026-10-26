import { z } from "zod";

import type { ExistingModel, SyncedFullModel, SyncProvider } from "../index.js";
import { factorBaseModel } from "./openrouter.js";

// `?currency=usd` quotes every route in USD, converting EUR-billed routes at
// LowRouter's monthly ECB fixing.
const API_ENDPOINT = "https://api.lowrouter.ai/v1/models?currency=usd";

// The feed lists over 200 chat auto IDs. Missing files are kept either way, but a
// short list that still parses is a truncated response, not a catalog to sync.
const MIN_LIVE_MODELS = 100;

// The price block of a header ends on the debit line, in the synced header and
// in the hand-written ones before it. Everything after it is evidence and is kept.
const SYNCED_SOURCE = `# Source: ${API_ENDPOINT} (synced).`;
const DEBITED = "# response carries the amount debited.";

// For a header with no debit line: where hand-written evidence (reasoning probes,
// limits, tool calls) starts. `Toggle` and `Budget` are the wire-path lines
// AGENTS.md requires on toggle and budget controls. `Anthropic's` is excluded:
// "# Anthropic's API, global endpoint;" names a default route inside the price
// block, as "# OpenAI's API" does.
const EVIDENCE = /^# (Toggle|Budget|Effort|Reasoning|Limit|Tool calls|Long-context|Anthropic(?!'s))\b/;

const Pricing = z.object({
  currency: z.string(),
  prompt_per_1m_tokens: z.number().nonnegative(),
  completion_per_1m_tokens: z.number().nonnegative(),
  cache_read_per_1m_tokens: z.number().nonnegative().optional(),
  cache_write_per_1m_tokens: z.number().nonnegative().optional(),
}).passthrough();

// The conversion LowRouter applied to a route billed in another currency.
const Fx = z.object({
  pair: z.string().length(6),
  rate: z.number().positive(),
  as_of: z.string(),
  source: z.string().optional(),
  fixing: z.string().optional(),
}).passthrough();

const Conversion = z.array(Fx).optional();

const LowRouterEntry = z.object({
  id: z.string().min(1),
  modality: z.string().optional(),
  lifecycle: z.string().optional(),
  endpoints: z.array(z.string()).optional(),
  auto_routing: z.object({
    default_routes: z.array(z.string()).optional(),
  }).passthrough().optional(),
}).passthrough();

// Only `data` is checked here: an entry with no string `id` is neither a chat ID
// nor a route's pinned entry, so it is skipped rather than failing the run.
export const LowRouterResponse = z.object({
  data: z.array(z.unknown()).transform((entries) => entries.filter(
    (entry): entry is { id: string } => typeof (entry as { id?: unknown } | null)?.id === "string",
  )),
}).passthrough();

/** An `auto/<creator>/<model>` ID with its default route's quote. */
export interface LowRouterModel {
  id: string;
  deprecated: boolean;
  route?: string;
  pricing?: z.infer<typeof Pricing>;
  /** The currency conversion behind the quote; empty for a route billed in USD. */
  fx?: z.infer<typeof Fx>[];
}

export const lowrouter = {
  id: "lowrouter",
  name: "LowRouter",
  modelsDir: "providers/lowrouter/models",
  // Absence is not retirement: LowRouter flags retiring IDs as deprecated, and
  // an ID can leave the chat listing while still served elsewhere.
  deleteMissing: false,
  // The feed publishes per-route controls, but a relay's reasoning metadata is
  // reviewed by hand, so new IDs are authored in batches rather than created.
  skipCreates: true,
  // Hundreds of IDs have no file yet; an issue each would flood the tracker
  // and hand LowRouter's reasoning metadata to the issue-fixer agent.
  trackMissingModels: false,
  authoritativeHeaders: true,
  sourceID(model) {
    return model.id;
  },
  skippedNotice(ids) {
    if (ids.length === 0) return [];
    const byCreator = new Map<string, number>();
    for (const id of ids) {
      const creator = id.split("/")[1]!;
      byCreator.set(creator, (byCreator.get(creator) ?? 0) + 1);
    }
    const counts = [...byCreator].sort(([a], [b]) => a.localeCompare(b)).map(([creator, count]) => `${creator} ${count}`);
    return [`${ids.length} LowRouter auto IDs have no file here and were not created (${counts.join(", ")}).`];
  },
  missingNotice(paths) {
    if (paths.length === 0) return [];
    return [`Kept ${paths.length} LowRouter files absent from the chat listing: ${paths.map((file) => `\`${file}\``).join(", ")}`];
  },
  async fetchModels() {
    const response = await fetch(API_ENDPOINT);
    if (!response.ok) {
      throw new Error(`LowRouter models request failed: ${response.status} ${response.statusText}`);
    }
    return response.json();
  },
  parseModels(raw) {
    return lowRouterModels(LowRouterResponse.parse(raw).data);
  },
  translateModel(model, context) {
    const existing = context.existing(model.id);
    if (existing === undefined) return undefined;
    return { id: model.id, ...buildLowRouterModel(model, existing, context.header?.(model.id) ?? "") };
  },
} satisfies SyncProvider<LowRouterModel>;

// Chat auto IDs served on Chat Completions, each priced at its first default
// route. Pinned entries (`<host>/<creator>/<model>`) carry the per-region quotes.
export function lowRouterModels(entries: Array<{ id: string }>): LowRouterModel[] {
  const pinned = new Map(entries.map((entry) => [entry.id, entry]));
  const models = entries
    // Only chat entries are parsed, so a malformed embedding or realtime entry
    // cannot fail the run.
    .filter((entry) => entry.id.startsWith("auto/") && entry.id.split("/").length === 3
      && ((entry as { modality?: unknown }).modality ?? "chat") === "chat")
    // A malformed entry is dropped like an unlisted one; a file behind it is
    // kept and named in the notices.
    .flatMap((entry) => LowRouterEntry.safeParse(entry).data ?? [])
    .filter((entry) => (entry.endpoints ?? []).includes("chat.completions"))
    .map((entry) => {
      const route = entry.auto_routing?.default_routes?.[0];
      const cut = route?.lastIndexOf("/") ?? -1;
      const host = route === undefined || cut === -1 ? undefined : pinned.get(route.slice(0, cut));
      const quote = host === undefined ? undefined : routeQuote(host, route!.slice(cut + 1));
      const pricing = quote?.pricing;
      if (pricing !== undefined && pricing.currency.toUpperCase() !== "USD") {
        throw new Error(`LowRouter ${entry.id}: expected a USD quote from ${API_ENDPOINT}, got ${pricing.currency}`);
      }
      return { id: entry.id, deprecated: entry.lifecycle === "deprecated", route, pricing, fx: quote?.fx };
    });
  if (models.length < MIN_LIVE_MODELS) {
    throw new Error(
      `LowRouter listed ${models.length} chat models, below the ${MIN_LIVE_MODELS} needed to`
        + " tell a catalog from a truncated response",
    );
  }
  return models;
}

// The quote of one region of a pinned `<host>/<creator>/<model>` entry, with the
// entry's conversion. Only that region and `fx` are parsed, so a malformed
// sibling region cannot void it. A malformed quote counts as none: a file behind
// it keeps its cost with a notice, and an ID with no file here cannot fail the run.
function routeQuote(host: object, locode: string) {
  const { regions, fx } = host as { regions?: unknown; fx?: unknown };
  const region = Array.isArray(regions)
    ? regions.find((item) => (item as { locode?: unknown } | null)?.locode === locode) as { pricing?: unknown } | undefined
    : undefined;
  const pricing = Pricing.safeParse(region?.pricing);
  const conversion = Conversion.safeParse(fx);
  return pricing.success && conversion.success ? { pricing: pricing.data, fx: conversion.data } : undefined;
}

// The default route's quote and LowRouter's deprecation flag are synced;
// reasoning controls, capabilities and limits stay hand-authored.
export function buildLowRouterModel(model: LowRouterModel, existing: ExistingModel, header: string) {
  const { base_model: baseModel, base_model_omit: baseModelOmit, ...current } = existing;
  const { pricing } = model;
  // A route that quotes no cache price (OpenAI's publish no cache write) keeps
  // the authored one rather than losing it.
  const quote = pricing && {
    input: pricing.prompt_per_1m_tokens,
    output: pricing.completion_per_1m_tokens,
    cache_read: pricing.cache_read_per_1m_tokens ?? existing.cost?.cache_read,
    cache_write: pricing.cache_write_per_1m_tokens ?? existing.cost?.cache_write,
  };
  const moved = quote !== undefined
    && Object.entries(quote).some(([key, value]) => existing.cost?.[key as keyof typeof quote] !== value);
  const values = {
    ...current,
    cost: quote === undefined ? existing.cost : { ...existing.cost, ...quote },
    // LowRouter owns its lifecycle: an ID it stops flagging is no longer deprecated.
    status: model.deprecated ? "deprecated" : existing.status === "deprecated" ? undefined : existing.status,
  } as SyncedFullModel;
  return {
    model: baseModel === undefined ? values : factorBaseModel(baseModel, values, existing.limit, baseModelOmit),
    header: quote === undefined ? header : lowRouterHeader(header, lowRouterCostHeader(model), moved),
    notices: quote === undefined
      ? [`LowRouter ${model.id}: default route ${model.route ?? "(none)"} has no quote; kept the authored cost.`]
      : undefined,
  };
}

// The synced price block: the route the cost was read from when it was synced,
// and the conversion behind it, which AGENTS.md asks a converted price to note
// with its rate and date. A new rate moves the price, so it is never stale.
export function lowRouterCostHeader(model: LowRouterModel) {
  // Split as lowRouterModels does: the host is the first segment, the locode the last.
  const route = model.route ?? "";
  const host = route.slice(0, route.indexOf("/"));
  const locode = route.slice(route.lastIndexOf("/") + 1);
  const [fx] = model.fx ?? [];
  const via = [fx?.source?.toUpperCase(), fx?.fixing && `${fx.fixing} fixing`].filter(Boolean).join(" ");
  return [
    SYNCED_SOURCE,
    `# Cost is the quote of the route LowRouter's default routing profile resolved this ID to when the price`,
    `# was synced: ${host}, ${locode} endpoint;`,
    fx === undefined
      ? "# that route bills in USD, so this is its quote with no currency conversion."
      : `# that route bills in ${fx.pair.slice(0, 3)}; converted at ${fx.rate} ${fx.pair.slice(3)}/${fx.pair.slice(0, 3)}`
        + ` (${via === "" ? "" : `${via}, `}${fx.as_of}).`,
    "# Another route can serve a request (the key's routing policy, or a failover); usage.cost on the",
    DEBITED,
  ];
}

// An unchanged price keeps its header byte for byte, so a route that moves at
// the same price opens no change. A moved price gets the synced price block in
// place of the old one, above the hand-written evidence, which is kept.
export function lowRouterHeader(header: string, costHeader: string[], moved: boolean) {
  if (!moved) return header;
  const lines = header.trimEnd().split("\n");
  const end = lines.indexOf(DEBITED);
  const start = end === -1 ? lines.findIndex((line) => EVIDENCE.test(line)) : end + 1;
  return [...costHeader, ...(start === -1 ? [] : lines.slice(start))].join("\n") + "\n";
}
