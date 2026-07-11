import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, symlinkSync } from "node:fs";
import test from "node:test";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { parseSettingsJson, settingsAreConfirmed } from "./set-photonvision-pipeline.mjs";

test("runs main when invoked through an aliased scripts directory", (context) => {
  const temporaryDirectory = mkdtempSync(join(tmpdir(), "photonvision-setter-"));
  context.after(() => rmSync(temporaryDirectory, { recursive: true, force: true }));

  const scriptsDirectory = dirname(fileURLToPath(import.meta.url));
  const aliasedDirectory = join(temporaryDirectory, "scripts");
  symlinkSync(scriptsDirectory, aliasedDirectory, process.platform === "win32" ? "junction" : "dir");

  const result = spawnSync(process.execPath, [join(aliasedDirectory, "set-photonvision-pipeline.mjs")], {
    encoding: "utf8",
    timeout: 5000
  });

  assert.notEqual(result.status, 0, "aliased direct invocation exited silently");
  assert.match(result.stderr, /Usage: node scripts\/set-photonvision-pipeline[.]mjs/);
});

test("accepts a nonempty plain settings object", () => {
  const settings = parseSettingsJson('{"enabled":true,"nested":{"ids":[1,2]}}');
  assert.deepEqual(settings, { enabled: true, nested: { ids: [1, 2] } });
});

test("confirms nested settings from an independent decoded instance", () => {
  const expected = { nested: { ids: [1, 2], mode: { enabled: true } } };
  const confirmed = JSON.parse(JSON.stringify(expected));
  assert.equal(settingsAreConfirmed(expected, confirmed), true);
});

test("rejects a nested settings mismatch", () => {
  const expected = { nested: { ids: [1, 2] } };
  const confirmed = { nested: { ids: [1, 3] } };
  assert.equal(settingsAreConfirmed(expected, confirmed), false);
});

for (const [name, json] of [
  ["empty object", "{}"],
  ["array", "[]"],
  ["null", "null"]
]) {
  test(`rejects ${name} settings`, () => {
    assert.throws(() => parseSettingsJson(json));
  });
}
