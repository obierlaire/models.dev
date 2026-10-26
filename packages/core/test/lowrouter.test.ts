import { expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";

import type { ExistingModel } from "../src/sync/index.js";
import {
  buildLowRouterModel,
  lowrouter,
  lowRouterHeader,
  type LowRouterModel,
  lowRouterModels,
} from "../src/sync/providers/lowrouter.js";

const MODELS_DIR = path.join(import.meta.dirname, "..", "..", "..", "models");

// A real lab model with the reasoning support a test needs, picked at run time
// so no test depends on one model staying in the catalog.
function labModel(supported: boolean): string {
  for (const lab of ["openai", "mistral", "anthropic"]) {
    for (const file of readdirSync(path.join(MODELS_DIR, lab)).sort()) {
      if (!file.endsWith(".toml")) continue;
      const toml = Bun.TOML.parse(readFileSync(path.join(MODELS_DIR, lab, file), "utf8")) as {
        capabilities?: { reasoning?: { supported?: boolean } };
      };
      if (toml.capabilities?.reasoning?.supported === supported) return `${lab}/${file.slice(0, -".toml".length)}`;
    }
  }
  throw new Error(`no lab model in models/ has reasoning.supported = ${supported}`);
}

const REASONING_MODEL = labModel(true);
const PLAIN_MODEL = labModel(false);

function autoEntry(id: string, route: string, overrides: Record<string, unknown> = {}) {
  return {
    id,
    modality: "chat",
    lifecycle: "active",
    endpoints: ["chat.completions", "messages", "responses"],
    auto_routing: { default_routes: [route] },
    ...overrides,
  };
}

function pinnedEntry(id: string, locode: string, pricing: Record<string, unknown>) {
  return { id, modality: "chat", regions: [{ locode, pricing: { currency: "USD", ...pricing } }] };
}

// Enough priced auto IDs to pass the truncation guard.
const FILLER = Array.from({ length: 100 }, (_, index) => [
  autoEntry(`auto/filler/model-${index}`, `host/filler/model-${index}/eu`),
  pinnedEntry(`host/filler/model-${index}`, "eu", { prompt_per_1m_tokens: 1, completion_per_1m_tokens: 2 }),
]).flat();

function parse(entries: unknown[]) {
  return lowrouter.parseModels({ data: [...entries, ...FILLER] }).filter((model) => !model.id.startsWith("auto/filler/"));
}

function model(overrides: Partial<LowRouterModel> = {}): LowRouterModel {
  return {
    id: "auto/testlab/test-model",
    deprecated: false,
    route: "somehost/testlab/test-model/eu",
    pricing: { currency: "USD", prompt_per_1m_tokens: 0.5282, completion_per_1m_tokens: 1.5845, cache_read_per_1m_tokens: 0.0528 },
    ...overrides,
  };
}

function existing(base: string, overrides: Record<string, unknown> = {}): ExistingModel {
  return { base_model: base, cost: { input: 0.5229, output: 1.5688, cache_read: 0.0523 }, ...overrides } as ExistingModel;
}

const HEADER = [
  "# Source: https://api.lowrouter.ai/v1/models (auto_routing, read 2026-10-09)",
  "# Cost: highest EUR price across those routes, converted at 1.1186 USD/EUR.",
  "# Effort: reasoning_effort = none|high, forwarded unchanged.",
  "# Limit: verified live.",
  "",
].join("\n");

test("lists chat auto IDs served on Chat Completions, priced at their first default route", () => {
  const models = parse([
    autoEntry("auto/mistralai/mistral-large-2512", "mistral/mistralai/mistral-large-2512/eu"),
    autoEntry("auto/mistralai/mistral-large-2512/eu", "mistral/mistralai/mistral-large-2512/eu"),
    autoEntry("auto/mistralai/mistral-embed-2312", "mistral/mistralai/mistral-embed-2312/eu", { modality: "embedding" }),
    autoEntry("auto/openai/gpt-5-pro", "openai/openai/gpt-5-pro/global", { endpoints: ["responses"] }),
    autoEntry("auto/openai/o3", "openai/openai/o3/global", { lifecycle: "deprecated" }),
    {
      id: "mistral/mistralai/mistral-large-2512",
      regions: [
        { locode: "global", pricing: { currency: "USD", prompt_per_1m_tokens: 0.4802, completion_per_1m_tokens: 1.4405 } },
        { locode: "eu", pricing: { currency: "USD", prompt_per_1m_tokens: 0.5282, completion_per_1m_tokens: 1.5845 } },
      ],
    },
    pinnedEntry("openai/openai/o3", "global", { prompt_per_1m_tokens: 2, completion_per_1m_tokens: 8 }),
  ]);
  expect(models).toEqual([
    {
      id: "auto/mistralai/mistral-large-2512",
      deprecated: false,
      route: "mistral/mistralai/mistral-large-2512/eu",
      pricing: expect.objectContaining({ prompt_per_1m_tokens: 0.5282, completion_per_1m_tokens: 1.5845 }),
    },
    {
      id: "auto/openai/o3",
      deprecated: true,
      route: "openai/openai/o3/global",
      pricing: expect.objectContaining({ prompt_per_1m_tokens: 2, completion_per_1m_tokens: 8 }),
    },
  ]);
});

test("a default route with no quote is listed unpriced", () => {
  const [unpriced] = parse([autoEntry("auto/testlab/test-model", "gone/testlab/test-model/eu")]);
  expect(unpriced).toEqual({ id: "auto/testlab/test-model", deprecated: false, route: "gone/testlab/test-model/eu", pricing: undefined });
});

test("a non-USD quote fails the run instead of publishing a wrong number", () => {
  expect(() => parse([
    autoEntry("auto/testlab/test-model", "somehost/testlab/test-model/eu"),
    { id: "somehost/testlab/test-model", regions: [{ locode: "eu", pricing: { currency: "EUR", prompt_per_1m_tokens: 0.4675, completion_per_1m_tokens: 1.4025 } }] },
  ])).toThrow("expected a USD quote");
});

test("refuses a response too short to be the whole catalog", () => {
  expect(() => lowrouter.parseModels({ data: FILLER.slice(0, 40) })).toThrow("truncated response");
});

test("refreshes the price from the default route and keeps authored reasoning", () => {
  const reasoning = [{ type: "effort", values: ["none", "high"] }];
  const built = buildLowRouterModel(model(), existing(REASONING_MODEL, { reasoning: true, reasoning_options: reasoning }), HEADER);
  expect(built.model).toMatchObject({
    base_model: REASONING_MODEL,
    reasoning_options: reasoning,
    cost: { input: 0.5282, output: 1.5845, cache_read: 0.0528 },
  });
  expect(built.notices).toBeUndefined();
});

test("keeps an authored cache price the route does not quote, and the header with an unchanged price", () => {
  const built = buildLowRouterModel(
    model({ pricing: { currency: "USD", prompt_per_1m_tokens: 4, completion_per_1m_tokens: 20, cache_read_per_1m_tokens: 0.4 } }),
    existing(PLAIN_MODEL, { cost: { input: 4, output: 20, cache_read: 0.4, cache_write: 5 } }),
    HEADER,
  );
  expect(built.model.cost).toMatchObject({ input: 4, output: 20, cache_read: 0.4, cache_write: 5 });
  expect(built.header).toBe(HEADER);
});

test("keeps context tiers authored under the base price", () => {
  const tiers = [{ tier: { type: "context", size: 272_000 }, input: 4, output: 15 }];
  const built = buildLowRouterModel(model(), existing(PLAIN_MODEL, { cost: { input: 2, output: 10, tiers } }), HEADER);
  expect(built.model.cost).toMatchObject({ input: 0.5282, tiers });
});

test("LowRouter's deprecated lifecycle sets status; otherwise the authored status stays", () => {
  const deprecated = buildLowRouterModel(model({ deprecated: true }), existing(PLAIN_MODEL), HEADER);
  expect(deprecated.model.status).toBe("deprecated");
  const beta = buildLowRouterModel(model(), existing(PLAIN_MODEL, { status: "beta" }), HEADER);
  expect(beta.model.status).toBe("beta");
});

test("an unpriced route keeps the authored cost and says so", () => {
  const built = buildLowRouterModel(model({ pricing: undefined, deprecated: true }), existing(PLAIN_MODEL), HEADER);
  expect(built.model).toMatchObject({ status: "deprecated", cost: { input: 0.5229, output: 1.5688, cache_read: 0.0523 } });
  expect(built.header).toBe(HEADER);
  expect(built.notices).toEqual([
    "LowRouter auto/testlab/test-model: default route somehost/testlab/test-model/eu has no quote; kept the authored cost.",
  ]);
});

test("a moved price replaces the cost lines and keeps the hand-written evidence", () => {
  const header = buildLowRouterModel(model(), existing(PLAIN_MODEL), HEADER).header;
  expect(header).toStartWith("# Source: https://api.lowrouter.ai/v1/models?currency=usd (synced).\n");
  expect(header).not.toContain("highest EUR price");
  expect(header).toEndWith("# Effort: reasoning_effort = none|high, forwarded unchanged.\n# Limit: verified live.\n");
  expect(lowRouterHeader(header, true)).toBe(header);
  expect(lowRouterHeader("# Cost: old route only.\n", true)).not.toContain("old route");
});

test("translateModel reads the header the runner passes in", () => {
  const file = existing(PLAIN_MODEL, { cost: { input: 0.5282, output: 1.5845, cache_read: 0.0528 } });
  const context = { existing: () => file, authored: () => file, header: () => HEADER };
  expect(lowrouter.translateModel(model(), context)).toMatchObject({ id: "auto/testlab/test-model", header: HEADER });
});

test("IDs without a file are skipped and summarized by creator", () => {
  const context = { existing: () => undefined, authored: () => undefined };
  expect(lowrouter.translateModel(model(), context)).toBeUndefined();
  expect(lowrouter.skippedNotice(["auto/qwen/a", "auto/google/b", "auto/qwen/c"])).toEqual([
    "3 LowRouter auto IDs have no file here and were not created (google 1, qwen 2).",
  ]);
});
