import { decode } from "@msgpack/msgpack";

const host = process.argv[2] ?? "127.0.0.1:5800";
const durationSeconds = Number(process.argv[3] ?? 12);
const warmupSeconds = Number(process.argv[4] ?? 2);

if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
  throw new Error("Duration must be a positive number of seconds");
}
if (!Number.isFinite(warmupSeconds) || warmupSeconds < 0) {
  throw new Error("Warmup must be a non-negative number of seconds");
}

const url = `ws://${host}/websocket_data`;
const socket = new WebSocket(url);
socket.binaryType = "arraybuffer";

const cameras = new Map();
const samples = new Map();
const updateCounts = new Map();
const staleUpdateCounts = new Map();
const invalidUpdateCounts = new Map();
const lastSequenceIds = new Map();
let collecting = false;

const percentile = (sorted, fraction) => {
  if (sorted.length === 0) return null;
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];
};

const average = (values) =>
  values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;

const summarize = (cameraSamples) => {
  const fps = cameraSamples.map((sample) => sample.fps).sort((a, b) => a - b);
  const latency = cameraSamples.map((sample) => sample.latency).sort((a, b) => a - b);
  const targetSamples = cameraSamples.filter((sample) => sample.targetCount > 0);
  const multitagSamples = cameraSamples.filter((sample) => sample.hasMultitag);
  const ids = [...new Set(cameraSamples.flatMap((sample) => sample.fiducialIds))].sort((a, b) => a - b);

  return {
    samples: cameraSamples.length,
    fps: {
      average: average(fps),
      minimum: fps[0] ?? null,
      p50: percentile(fps, 0.5),
      p95: percentile(fps, 0.95),
      maximum: fps.at(-1) ?? null
    },
    latencyMs: {
      average: average(latency),
      minimum: latency[0] ?? null,
      p50: percentile(latency, 0.5),
      p95: percentile(latency, 0.95),
      maximum: latency.at(-1) ?? null
    },
    targetHitRate: cameraSamples.length === 0 ? null : targetSamples.length / cameraSamples.length,
    averageTargets: average(cameraSamples.map((sample) => sample.targetCount)),
    multitagHitRate: cameraSamples.length === 0 ? null : multitagSamples.length / cameraSamples.length,
    fiducialIdsSeen: ids
  };
};

const finish = () => {
  const result = {};
  const uniqueNames = new Set([...cameras.keys(), ...samples.keys()]);
  for (const uniqueName of uniqueNames) {
    const cameraSamples = samples.get(uniqueName) ?? [];
    result[uniqueName] = {
      camera: cameras.get(uniqueName) ?? null,
      updates: updateCounts.get(uniqueName) ?? 0,
      staleUpdates: staleUpdateCounts.get(uniqueName) ?? 0,
      invalidUpdates: invalidUpdateCounts.get(uniqueName) ?? 0,
      ...summarize(cameraSamples)
    };
  }

  console.log(JSON.stringify({ url, durationSeconds, warmupSeconds, result }, null, 2));
  socket.close();
};

socket.addEventListener("open", () => {
  setTimeout(() => {
    collecting = true;
    setTimeout(finish, durationSeconds * 1000);
  }, warmupSeconds * 1000);
});

socket.addEventListener("message", (event) => {
  const message = decode(new Uint8Array(event.data));

  for (const camera of message.cameraSettings ?? []) {
    cameras.set(camera.uniqueName, {
      nickname: camera.nickname,
      uniqueName: camera.uniqueName,
      videoModeIndex: camera.currentPipelineSettings?.cameraVideoModeIndex,
      videoMode: camera.videoFormatList?.[camera.currentPipelineSettings?.cameraVideoModeIndex],
      pipelineType: camera.currentPipelineSettings?.pipelineType,
      decimate: camera.currentPipelineSettings?.decimate,
      blockForFrames: camera.currentPipelineSettings?.blockForFrames,
      streamingFrameDivisor: camera.currentPipelineSettings?.streamingFrameDivisor,
      inputShouldShow: camera.currentPipelineSettings?.inputShouldShow,
      outputShouldShow: camera.currentPipelineSettings?.outputShouldShow,
      useCudaTagDetection: camera.currentPipelineSettings?.useCudaTagDetection,
      useMLDetection: camera.currentPipelineSettings?.useMLDetection,
      cameraAutoExposure: camera.currentPipelineSettings?.cameraAutoExposure,
      solvePNPEnabled: camera.currentPipelineSettings?.solvePNPEnabled,
      doMultiTarget: camera.currentPipelineSettings?.doMultiTarget
    });
  }

  if (!collecting || message.updatePipelineResult === undefined) return;

  for (const [uniqueName, result] of Object.entries(message.updatePipelineResult)) {
    if (!samples.has(uniqueName)) samples.set(uniqueName, []);
    updateCounts.set(uniqueName, (updateCounts.get(uniqueName) ?? 0) + 1);

    if (lastSequenceIds.get(uniqueName) === result.sequenceID) {
      staleUpdateCounts.set(uniqueName, (staleUpdateCounts.get(uniqueName) ?? 0) + 1);
      continue;
    }
    lastSequenceIds.set(uniqueName, result.sequenceID);

    if (result.sequenceID <= 0 || result.fps <= 0) {
      invalidUpdateCounts.set(uniqueName, (invalidUpdateCounts.get(uniqueName) ?? 0) + 1);
      continue;
    }

    samples.get(uniqueName).push({
      fps: result.fps,
      latency: result.latency,
      targetCount: result.targets.length,
      hasMultitag: result.multitagResult !== undefined,
      fiducialIds: result.targets.map((target) => target.fiducialId).filter((id) => id >= 0)
    });
  }
});

socket.addEventListener("error", () => {
  throw new Error(`Could not connect to ${url}`);
});
