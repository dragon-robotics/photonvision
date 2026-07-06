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
  const { availableModels, supportedBackends } = useSettingsStore().general;
  const rubikSupported = supportedBackends.some((backend: string) => backend.toLowerCase() === "rubik");
  if (!rubikSupported) return [];

  return availableModels.filter(
    (model: ObjectDetectionModelProperties) =>
      model.family.toLowerCase() === "rubik" && model.nickname.toLowerCase().includes("apriltag")
  );
});

const mlDetectionAvailable = computed(() => supportedAprilTagModels.value.length > 0);

const selectedAprilTagModel = computed({
  get: () => {
    const currentModel = currentPipelineSettings.value.model;
    if (!currentModel) return undefined;

    const index = supportedAprilTagModels.value.findIndex((model) => model.modelPath === currentModel.modelPath);
    return index === -1 ? undefined : index;
  },

  set: (value) => {
    if (value !== undefined && value >= 0 && value < supportedAprilTagModels.value.length) {
      useCameraSettingsStore().changeCurrentPipelineSetting({ model: supportedAprilTagModels.value[value] }, true);
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
    <v-divider v-if="mlDetectionAvailable" class="mt-3 mb-2" />
    <div v-if="mlDetectionAvailable">
      <p class="text-subtitle-2 mb-2">ML-Tag</p>
      <pv-switch
        v-model="currentPipelineSettings.useMLDetection"
        :switch-cols="interactiveCols"
        label="Enable ML-Tag"
        tooltip="Uses the Rubik Pi 3 NPU to find AprilTag regions before decoding tags."
        @update:modelValue="
          (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ useMLDetection: value }, false)
        "
      />
      <div v-if="currentPipelineSettings.useMLDetection">
        <pv-select
          v-model="selectedAprilTagModel"
          label="Model"
          tooltip="The Rubik model used to find AprilTag regions."
          :select-cols="interactiveCols"
          :items="supportedAprilTagModels.map((model) => model.nickname)"
        />
        <pv-slider
          v-model="currentPipelineSettings.mlConfidenceThreshold"
          :slider-cols="interactiveCols"
          label="Confidence"
          tooltip="Minimum ML confidence for ROI detection."
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
          tooltip="Overlap threshold used to merge ML ROI detections."
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
          tooltip="Pixels added around each detected AprilTag region."
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
          tooltip="Draws the ML ROI boxes on the processed stream."
          @update:modelValue="
            (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ showDetectionBoxes: value }, false)
          "
        />
      </div>
    </div>
  </div>
</template>
