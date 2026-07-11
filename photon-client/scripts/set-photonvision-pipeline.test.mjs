import assert from "node:assert/strict";
import test from "node:test";

import { parseSettingsJson, settingsAreConfirmed } from "./set-photonvision-pipeline.mjs";

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
