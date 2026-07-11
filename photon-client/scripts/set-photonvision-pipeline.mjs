import { decode, encode } from "@msgpack/msgpack";
import { pathToFileURL } from "node:url";
import { isDeepStrictEqual } from "node:util";

export const parseSettingsJson = (settingsJson) => {
  const settings = JSON.parse(settingsJson);
  if (
    settings === null ||
    Array.isArray(settings) ||
    typeof settings !== "object" ||
    Object.getPrototypeOf(settings) !== Object.prototype
  ) {
    throw new Error("Settings JSON must be a plain object");
  }
  if (Object.keys(settings).length === 0) {
    throw new Error("Settings JSON object must not be empty");
  }
  return settings;
};

export const settingsAreConfirmed = (expectedSettings, confirmedSettings) =>
  Object.entries(expectedSettings).every(([key, value]) =>
    isDeepStrictEqual(confirmedSettings[key], value)
  );

const main = () => {
  const [host, cameraUniqueName, settingsJson] = process.argv.slice(2);

  if (!host || !cameraUniqueName || !settingsJson) {
    throw new Error(
      "Usage: node scripts/set-photonvision-pipeline.mjs <host:port> <camera-unique-name> '<settings-json>'"
    );
  }

  const settings = parseSettingsJson(settingsJson);
  const socket = new WebSocket(`ws://${host}/websocket_data`);
  socket.binaryType = "arraybuffer";
  const confirmedSettings = {};
  let verificationSocket;

  const timeout = setTimeout(() => {
    socket.close();
    verificationSocket?.close();
    throw new Error("Timed out waiting for PhotonVision to confirm the pipeline update");
  }, 5000);

  const finishIfConfirmed = () => {
    if (!settingsAreConfirmed(settings, confirmedSettings)) return false;

    clearTimeout(timeout);
    console.log(JSON.stringify({ cameraUniqueName, confirmed: settings }, null, 2));
    socket.close();
    verificationSocket?.close();
    return true;
  };

  const verifyWithFreshConnection = () => {
    verificationSocket = new WebSocket(`ws://${host}/websocket_data`);
    verificationSocket.binaryType = "arraybuffer";
    verificationSocket.addEventListener("message", (event) => {
      const message = decode(new Uint8Array(event.data));
      const camera = (message.cameraSettings ?? []).find(
        (candidate) => candidate.uniqueName === cameraUniqueName
      );
      if (camera === undefined) return;

      Object.assign(confirmedSettings, camera.currentPipelineSettings);
      finishIfConfirmed();
    });
    verificationSocket.addEventListener("error", () => {
      clearTimeout(timeout);
      throw new Error(`Could not verify ws://${host}/websocket_data with a fresh connection`);
    });
  };

  socket.addEventListener("open", () => {
    socket.send(
      encode({
        changePipelineSetting: {
          ...settings,
          cameraUniqueName
        }
      })
    );
    setTimeout(verifyWithFreshConnection, 500);
  });

  socket.addEventListener("error", () => {
    clearTimeout(timeout);
    throw new Error(`Could not connect to ws://${host}/websocket_data`);
  });
};

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
