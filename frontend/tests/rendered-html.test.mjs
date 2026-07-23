import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the TorchForge workspace", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>TorchForge — Paper to PyTorch<\/title>/i);
  assert.match(html, /Forge research into/);
  assert.match(html, /Drop a research paper here/);
  assert.match(html, /Paper library/);
  assert.doesNotMatch(html, /Your site is taking shape|codex-preview/i);
});

test("ships real pipeline interactions instead of starter UI", async () => {
  const [app, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/torchforge-app.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(app, /\/api\/papers/);
  assert.match(app, /runStage/);
  assert.match(app, /openArtifact/);
  assert.match(app, /application\/pdf/);
  assert.match(layout, /og\.png/);
  assert.match(packageJson, /"name": "torchforge-studio"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
