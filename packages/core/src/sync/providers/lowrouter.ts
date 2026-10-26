import { z } from "zod";

import type { ExistingModel, SyncedFullModel, SyncProvider } from "../index.js";
import { factorBaseModel } from "./openrouter.js";

// `?currency=usd` quotes every route in USD, converting EUR-billed routes at
// LowRouter's monthly ECB fixing.
const API_ENDPOINT = "https://api.lowrouter.ai/v1/models?currency=usd";

// The feed lists over 200 chat auto IDs. Missing files are kept either way, but a
// short list that still parses is a truncated response, not a catalog to sync.
const MIN_LIVE_MODELS = 100;

// Where hand-written evidence (reasoning probes, limits, tool calls) starts in a
// header. Everything above it describes the price and is replaced when it moves.
// `Toggle` and `Budget` are the wire-path lines AGENTS.md requires on toggle and
// budget controls. `Anthropic's` is excluded: "# Anthropic's API, global endpoint;"
// names a default route inside the price block, as "# OpenAI's API" does.
const EVIDENCE = /^# (Toggle|Budget|Effort|Reasoning|Limit|Tool calls|Long-context|Anthropic(?!'s))\b/;

const COST_HEADER = [
  `# Source: ${API_ENDPOINT} (synced).`,
  "# Cost is the quote of the route LowRouter's default routing profile resolves this ID to; EUR-billed",
  "# routes are converted at LowRouter's monthly ECB fixing. Another route can serve a request (the key's",
  "# routing policy, or a failover); usage.cost on the response carries the amount debited.",
];

const Pricing = z.object({
  currency: z.string(),
  prompt_per_1m_tokens: z.number().nonnegative(),
  completion_per_1m_tokens: z.number().nonnegative(),
  cache_read_per_1m_tokens: z.number().nonnegative().optional(),
  cache_write_per_1m_tokens: z.number().nonnegative().optional(),
}).passthrough();

const LowRouterEntry = z.object({
  id: z.string().min(1),
  modality: z.string().optional(),
  lifecycle: z.string().optional(),
  endpoints: z.array(z.string()).optional(),
  auto_routing: z.object({
    default_routes: z.array(z.string()).optional(),
  }).passthrough().optional(),
  regions: z.array(z.object({
    locode: z.string(),
    pricing: Pricing.optional(),
  }).passthrough()).optional(),
}).passthrough();

export const LowRouterResponse = z.object({
  data: z.array(LowRouterEntry),
}).passthrough();

type LowRouterEntry = z.infer<typeof LowRouterEntry>;

/** An `auto/<creator>/<model>` ID with its default route's quote. */
export interface LowRouterModel {
  id: string;
  deprecated: boolean;
  route?: string;
  pricing?: z.infer<typeof Pricing>;
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
export function lowRouterModels(entries: LowRouterEntry[]): LowRouterModel[] {
  const pinned = new Map(entries.map((entry) => [entry.id, entry]));
  const models = entries
    .filter((entry) =>
      entry.id.startsWith("auto/")
      && entry.id.split("/").length === 3
      && (entry.modality ?? "chat") === "chat"
      && (entry.endpoints ?? []).includes("chat.completions"))
    .map((entry) => {
      const route = entry.auto_routing?.default_routes?.[0];
      const cut = route?.lastIndexOf("/") ?? -1;
      const pricing = route === undefined || cut === -1
        ? undefined
        : pinned.get(route.slice(0, cut))?.regions?.find((region) => region.locode === route.slice(cut + 1))?.pricing;
      if (pricing !== undefined && pricing.currency.toUpperCase() !== "USD") {
        throw new Error(`LowRouter ${entry.id}: expected a USD quote from ${API_ENDPOINT}, got ${pricing.currency}`);
      }
      return { id: entry.id, deprecated: entry.lifecycle === "deprecated", route, pricing };
    });
  if (models.length < MIN_LIVE_MODELS) {
    throw new Error(
      `LowRouter listed ${models.length} chat models, below the ${MIN_LIVE_MODELS} needed to`
        + " tell a catalog from a truncated response",
    );
  }
  return models;
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
    header: lowRouterHeader(header, moved),
    notices: quote === undefined
      ? [`LowRouter ${model.id}: default route ${model.route ?? "(none)"} has no quote; kept the authored cost.`]
      : undefined,
  };
}

// An unchanged price keeps its header byte for byte. A moved one gets the
// synced cost lines in place of whatever described the old price, above the
// hand-written evidence, which is kept.
export function lowRouterHeader(header: string, moved: boolean) {
  if (!moved) return header;
  const lines = header.trimEnd().split("\n");
  const start = lines.findIndex((line) => EVIDENCE.test(line));
  return [...COST_HEADER, ...(start === -1 ? [] : lines.slice(start))].join("\n") + "\n";
}
