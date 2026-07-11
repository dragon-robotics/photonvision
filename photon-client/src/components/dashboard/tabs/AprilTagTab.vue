<script setup lang="ts">
import { type AprilTagPipelineSettings, PipelineType } from "@/types/PipelineTypes";
import PvSelect from "@/components/common/pv-select.vue";
import PvSlider from "@/components/common/pv-slider.vue";
import PvSwitch from "@/components/common/pv-switch.vue";
import { computed } from "vue";
import { useStateStore } from "@/stores/StateStore";
import { useCameraSettingsStore } from "@/stores/settings/CameraSettingsStore";
import { useSettingsStore } from "@/stores/settings/GeneralSettingsStore";
import type { ObjectDetectionModelProperties } from "@/types/SettingTypes";
import { useDisplay } from "vuetify";

// TODO fix pipeline typing in order to fix this, the store settings call should be able to infer that only valid pipeline type settings are exposed based on pre-checks for the entire config section
// Defer reference to store access method
const currentPipelineSettings = computed<AprilTagPipelineSettings>(
  () => useCameraSettingsStore().currentPipelineSettings as AprilTagPipelineSettings
);
const { mdAndDown } = useDisplay();
const interactiveCols = computed(() =>
  mdAndDown.value && (!useStateStore().sidebarFolded || useCameraSettingsStore().isDriverMode) ? 8 : 7
);

const supportedAprilTagModels = computed<ObjectDetectionModelProperties[]>(() => {
  const { availableModels } = useSettingsStore().general;

  return availableModels.filter(
    (model: ObjectDetectionModelProperties) =>
      model.family === "TENSORRT" && model.labels.some((label) => label.toLowerCase() === "apriltag")
  );
});

const mlDetectionAvailable = computed(() => useSettingsStore().general.supportedBackends.includes("TENSORRT"));

const selectedAprilTagModel = computed({
  get: () => {
    const currentModelName = currentPipelineSettings.value.mlModelName;
    if (!currentModelName) return undefined;

    const index = supportedAprilTagModels.value.findIndex((model) => model.modelPath === currentModelName);
    return index === -1 ? undefined : index;
  },

  set: (value) => {
    if (value !== undefined && value >= 0 && value < supportedAprilTagModels.value.length) {
      useCameraSettingsStore().changeCurrentPipelineSetting(
        { mlModelName: supportedAprilTagModels.value[value].modelPath },
        true
      );
    }
  }
});
</script>

<template>
  <div v-if="currentPipelineSettings.pipelineType === PipelineType.AprilTag">
    <pv-select
      v-model="currentPipelineSettings.tagFamily"
      label="Target family"
      :items="['AprilTag 36h11 (6.5in)', 'AprilTag 16h5 (6in)']"
      :select-cols="interactiveCols"
      @update:modelValue="(value) => useCameraSettingsStore().changeCurrentPipelineSetting({ tagFamily: value }, false)"
    />
    <pv-slider
      v-model="currentPipelineSettings.decimate"
      :slider-cols="interactiveCols"
      label="Decimate"
      tooltip="Increases FPS at the expense of range by reducing image resolution initially"
      :min="1"
      :max="8"
      @update:modelValue="(value) => useCameraSettingsStore().changeCurrentPipelineSetting({ decimate: value }, false)"
    />
    <pv-slider
      v-model="currentPipelineSettings.blur"
      :slider-cols="interactiveCols"
      label="Blur"
      tooltip="Gaussian blur added to the image, high FPS cost for slightly decreased noise"
      :min="0"
      :max="5"
      :step="0.1"
      @update:modelValue="(value) => useCameraSettingsStore().changeCurrentPipelineSetting({ blur: value }, false)"
    />
    <pv-slider
      v-model="currentPipelineSettings.threads"
      :slider-cols="interactiveCols"
      label="Threads"
      tooltip="Number of threads spawned by the AprilTag detector"
      :min="1"
      :max="8"
      @update:modelValue="(value) => useCameraSettingsStore().changeCurrentPipelineSetting({ threads: value }, false)"
    />
    <pv-slider
      v-model="currentPipelineSettings.decisionMargin"
      :slider-cols="interactiveCols"
      label="Decision Margin Cutoff"
      tooltip="Tags with a 'margin' (decoding quality score) less than this wil be rejected. Increase this to reduce the number of false positive detections"
      :min="0"
      :max="250"
      @update:modelValue="
        (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ decisionMargin: value }, false)
      "
    />
    <pv-slider
      v-model="currentPipelineSettings.numIterations"
      :slider-cols="interactiveCols"
      label="Pose Estimation Iterations"
      tooltip="Number of iterations the pose estimation algorithm will run, 50-100 is a good starting point"
      :min="0"
      :max="500"
      @update:modelValue="
        (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ numIterations: value }, false)
      "
    />
    <pv-switch
      v-model="currentPipelineSettings.refineEdges"
      :switch-cols="interactiveCols"
      label="Refine Edges"
      tooltip="Further refines the AprilTag corner position initial estimate, suggested left on"
      @update:modelValue="
        (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ refineEdges: value }, false)
      "
    />
    <v-divider class="mt-3 mb-2" />
    <p class="text-subtitle-2 mb-2">GPU Detection</p>
    <pv-switch
      v-model="currentPipelineSettings.useCudaTagDetection"
      :switch-cols="interactiveCols"
      label="CUDA AprilTag Detector"
      tooltip="Runs full-frame AprilTag 36h11 detection on the Jetson GPU. Falls back automatically when unavailable."
      @update:modelValue="
        (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ useCudaTagDetection: value }, false)
      "
    />
    <v-divider v-if="mlDetectionAvailable" class="mt-3 mb-2" />
    <div v-if="mlDetectionAvailable">
      <p class="text-subtitle-2 mb-2">ML ROI Detection</p>
      <pv-switch
        v-model="currentPipelineSettings.useMLDetection"
        :switch-cols="interactiveCols"
        label="AI-Assisted ROI Detection (TensorRT)"
        tooltip="Uses TensorRT to find AprilTag regions before CPU decoding."
        @update:modelValue="
          (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ useMLDetection: value }, false)
        "
      />
      <div v-if="currentPipelineSettings.useMLDetection">
        <pv-select
          v-model="selectedAprilTagModel"
          label="Model"
          tooltip="The TensorRT model accelerated by the Orin GPU for AprilTag ROI detection."
          :select-cols="interactiveCols"
          :items="supportedAprilTagModels.map((model) => model.nickname)"
        />
        <pv-slider
          v-model="currentPipelineSettings.mlConfidenceThreshold"
          :slider-cols="interactiveCols"
          label="Confidence"
          tooltip="Minimum CUDA ROI confidence required before AprilTag decoding."
          :min="0"
          :max="1"
          :step="0.01"
          @update:modelValue="
            (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ mlConfidenceThreshold: value }, false)
          "
        />
        <pv-slider
          v-model="currentPipelineSettings.mlNmsThreshold"
          :slider-cols="interactiveCols"
          label="NMS Threshold"
          tooltip="Overlap threshold used by the Orin GPU detector to merge ROI candidates."
          :min="0"
          :max="1"
          :step="0.01"
          @update:modelValue="
            (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ mlNmsThreshold: value }, false)
          "
        />
        <pv-slider
          v-model="currentPipelineSettings.mlRoiPaddingPixels"
          :slider-cols="interactiveCols"
          label="ROI Padding"
          tooltip="Pixels added around each GPU-detected AprilTag region before decode."
          :min="0"
          :max="150"
          :step="5"
          @update:modelValue="
            (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ mlRoiPaddingPixels: value }, false)
          "
        />
        <pv-switch
          v-model="currentPipelineSettings.showDetectionBoxes"
          :switch-cols="interactiveCols"
          label="Show ROI Boxes"
          tooltip="Draws CUDA-detected ROI boxes on the processed stream."
          @update:modelValue="
            (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ showDetectionBoxes: value }, false)
          "
        />
      </div>
    </div>
  </div>
</template>
