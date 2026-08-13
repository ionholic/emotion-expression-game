const MEDIAPIPE_VERSION = "0.10.35";
const MEDIAPIPE_MODULE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/+esm`;
const MEDIAPIPE_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const ROUND_TIMES = {
  prepare: 3000,
  challenge: 10000,
  final: 5000
};

const FACE_STABLE_MS = 600;
const SUCCESS_HOLD_MS = 850;
const FINAL_SAMPLE_FROM_MS = 3000;
const DETECT_DURATION_MS = 5000;
const CLOCK_CIRCUMFERENCE = 214;
const MODEL_LOAD_TIMEOUT_MS = 30000;

const EMOTIONS = [
  {
    id: "joy",
    name: "기쁨",
    orderLabel: "첫 번째 마음",
    prompt: "좋아하는 사람을 만난 것처럼 환하게 웃어 보세요.",
    guide: {
      eyebrows: "힘을 빼고 자연스럽게 두어요.",
      eyes: "웃을 때 눈꼬리가 살짝 접혀요.",
      nose: "코 주변은 편안하게 두어요.",
      mouth: "양쪽 입꼬리를 위로 올려요."
    },
    threshold: 0.58,
    features: [
      { names: ["mouthSmileLeft", "mouthSmileRight"], target: 0.47, weight: 0.72 },
      { names: ["cheekSquintLeft", "cheekSquintRight"], target: 0.25, weight: 0.28 }
    ],
    penalties: [{ names: ["jawOpen"], target: 0.7, weight: 0.08 }]
  },
  {
    id: "sad",
    name: "슬픔",
    orderLabel: "두 번째 마음",
    prompt: "아끼던 물건을 잃어버린 것처럼 슬픈 얼굴을 지어 보세요.",
    guide: {
      eyebrows: "눈썹 안쪽을 살짝 위로 모아요.",
      eyes: "시선을 조금 아래로 내려요.",
      nose: "코 주변은 자연스럽게 두어요.",
      mouth: "입꼬리를 천천히 아래로 내려요."
    },
    threshold: 0.5,
    features: [
      { names: ["browInnerUp"], target: 0.34, weight: 0.5 },
      { names: ["mouthFrownLeft", "mouthFrownRight"], target: 0.32, weight: 0.5 }
    ]
  },
  {
    id: "anger",
    name: "화남",
    orderLabel: "세 번째 마음",
    prompt: "불공평한 일을 만난 것처럼 단단한 화난 얼굴을 만들어 보세요.",
    guide: {
      eyebrows: "눈썹을 아래로 내려 가운데로 모아요.",
      eyes: "눈에 살짝 힘을 주어 바라봐요.",
      nose: "콧방울에 조금 힘이 들어가도 좋아요.",
      mouth: "입술을 꾹 다물어 보세요."
    },
    threshold: 0.53,
    features: [
      { names: ["browDownLeft", "browDownRight"], target: 0.38, weight: 0.5 },
      { names: ["eyeSquintLeft", "eyeSquintRight"], target: 0.27, weight: 0.2 },
      { names: ["mouthPressLeft", "mouthPressRight"], target: 0.28, weight: 0.3 }
    ]
  },
  {
    id: "surprise",
    name: "놀람",
    orderLabel: "네 번째 마음",
    prompt: "생각하지 못한 선물을 받은 것처럼 깜짝 놀라 보세요.",
    guide: {
      eyebrows: "양쪽 눈썹을 위로 올려요.",
      eyes: "눈을 동그랗게 크게 떠요.",
      nose: "코 주변은 편안하게 두어요.",
      mouth: "입을 동그랗게 벌려요."
    },
    threshold: 0.56,
    features: [
      { names: ["eyeWideLeft", "eyeWideRight"], target: 0.38, weight: 0.3 },
      { names: ["browOuterUpLeft", "browOuterUpRight"], target: 0.34, weight: 0.24 },
      { names: ["jawOpen"], target: 0.48, weight: 0.46 }
    ]
  },
  {
    id: "fear",
    name: "무서움",
    orderLabel: "다섯 번째 마음",
    prompt: "어두운 방에서 작은 소리를 들은 것처럼 무서운 얼굴을 지어 보세요.",
    guide: {
      eyebrows: "눈썹 안쪽을 위로 모아요.",
      eyes: "눈을 크게 뜨고 주변을 살펴봐요.",
      nose: "코 주변에 살짝 힘이 들어가요.",
      mouth: "입을 옆으로 길게 당겨요."
    },
    threshold: 0.48,
    features: [
      { names: ["eyeWideLeft", "eyeWideRight"], target: 0.31, weight: 0.34 },
      { names: ["browInnerUp"], target: 0.28, weight: 0.29 },
      { names: ["mouthStretchLeft", "mouthStretchRight"], target: 0.3, weight: 0.37 }
    ],
    penalties: [{ names: ["mouthSmileLeft", "mouthSmileRight"], target: 0.7, weight: 0.12 }]
  },
  {
    id: "disgust",
    name: "싫음",
    orderLabel: "여섯 번째 마음",
    prompt: "마음에 들지 않는 냄새를 맡은 것처럼 얼굴을 찡그려 보세요.",
    guide: {
      eyebrows: "눈썹을 살짝 아래로 내려요.",
      eyes: "눈을 조금 가늘게 떠요.",
      nose: "코 주변을 위로 찡그려요.",
      mouth: "윗입술을 살짝 올려요."
    },
    threshold: 0.47,
    features: [
      { names: ["noseSneerLeft", "noseSneerRight"], target: 0.3, weight: 0.57 },
      { names: ["mouthUpperUpLeft", "mouthUpperUpRight"], target: 0.28, weight: 0.43 }
    ]
  },
  {
    id: "worry",
    name: "걱정",
    orderLabel: "일곱 번째 마음",
    prompt: "친구에게 무슨 일이 생겼을까 생각하는 걱정스러운 얼굴을 지어 보세요.",
    guide: {
      eyebrows: "눈썹 안쪽을 위로 모아요.",
      eyes: "시선을 한곳에 모아 생각해요.",
      nose: "코 주변은 자연스럽게 두어요.",
      mouth: "입술을 가볍게 눌러요."
    },
    threshold: 0.46,
    features: [
      { names: ["browInnerUp"], target: 0.3, weight: 0.62 },
      { names: ["mouthPressLeft", "mouthPressRight"], target: 0.23, weight: 0.38 }
    ],
    penalties: [{ names: ["mouthFrownLeft", "mouthFrownRight"], target: 0.8, weight: 0.08 }]
  },
  {
    id: "shy",
    name: "부끄러움",
    orderLabel: "여덟 번째 마음",
    prompt: "칭찬을 들었을 때처럼 수줍고 부끄러운 얼굴을 표현해 보세요.",
    guide: {
      eyebrows: "눈썹의 힘을 편안하게 풀어요.",
      eyes: "시선을 살짝 아래로 내려요.",
      nose: "코 주변은 자연스럽게 두어요.",
      mouth: "작게 미소 지어 보세요."
    },
    threshold: 0.43,
    features: [
      { names: ["eyeLookDownLeft", "eyeLookDownRight"], target: 0.25, weight: 0.42 },
      { names: ["mouthSmileLeft", "mouthSmileRight"], target: 0.26, weight: 0.35 },
      { names: ["mouthDimpleLeft", "mouthDimpleRight"], target: 0.2, weight: 0.23 }
    ]
  },
  {
    id: "excited",
    name: "신남",
    orderLabel: "아홉 번째 마음",
    prompt: "기다리던 소풍을 떠나는 것처럼 신나는 얼굴을 크게 표현해 보세요.",
    guide: {
      eyebrows: "눈썹을 조금 위로 올려요.",
      eyes: "눈을 반짝이듯 크게 떠요.",
      nose: "코 주변은 편안하게 두어요.",
      mouth: "활짝 웃으며 입도 열어 봐요."
    },
    threshold: 0.55,
    features: [
      { names: ["mouthSmileLeft", "mouthSmileRight"], target: 0.48, weight: 0.48 },
      { names: ["jawOpen"], target: 0.34, weight: 0.3 },
      { names: ["eyeWideLeft", "eyeWideRight"], target: 0.24, weight: 0.22 }
    ]
  },
  {
    id: "calm",
    name: "편안함",
    orderLabel: "열 번째 마음",
    prompt: "따뜻한 햇살을 느끼는 것처럼 얼굴의 힘을 편안히 풀어 보세요.",
    guide: {
      eyebrows: "눈썹의 힘을 천천히 풀어요.",
      eyes: "눈을 편안하게 뜨거나 감아요.",
      nose: "코로 천천히 숨을 쉬어요.",
      mouth: "입 주변의 힘을 가볍게 풀어요."
    },
    threshold: 0.68,
    customScore: scoreCalmExpression
  }
];

const state = {
  mode: "practice",
  journey: "game",
  screen: "welcome",
  stream: null,
  faceLandmarker: null,
  faceModelPromise: null,
  currentBlendshapes: null,
  baseline: {},
  featureScales: {},
  currentEmotionIndex: 0,
  results: [],
  rawScores: createEmptyScores(),
  smoothedScores: createEmptyScores(),
  finalSamples: [],
  roundPhase: "idle",
  phaseElapsed: 0,
  lastClockAt: 0,
  faceStableSince: 0,
  faceDetected: false,
  targetHoldStarted: 0,
  lastDetectionAt: 0,
  detectionInterval: 82,
  lastVideoTime: -1,
  animationFrameId: 0,
  cameraRequestId: 0,
  cameraStarting: false,
  sessionVersion: 0,
  pendingRoundVersion: 0,
  roundResultReturnFocus: null,
  successLocked: false,
  userPaused: false,
  autoPaused: false,
  detectPhase: "idle",
  detectElapsed: 0,
  detectLastAt: 0,
  detectSamples: [],
  detectResult: null,
  guidePartIndex: 0,
  lastGuideChangeAt: 0,
  soundEnabled: true,
  audioContext: null,
  toastTimer: 0
};

const ui = {
  welcomeScreen: document.getElementById("welcomeScreen"),
  setupScreen: document.getElementById("setupScreen"),
  gameScreen: document.getElementById("gameScreen"),
  detectScreen: document.getElementById("detectScreen"),
  resultScreen: document.getElementById("resultScreen"),
  sessionProgress: document.getElementById("sessionProgress"),
  sessionProgressText: document.getElementById("sessionProgressText"),
  sessionProgressBar: document.getElementById("sessionProgressBar"),
  soundButton: document.getElementById("soundButton"),
  soundLabel: document.querySelector(".sound-label"),
  cameraStartButton: document.getElementById("cameraStartButton"),
  detectStartButton: document.getElementById("detectStartButton"),
  practiceStartButton: document.getElementById("practiceStartButton"),
  setupPracticeButton: document.getElementById("setupPracticeButton"),
  retryCameraButton: document.getElementById("retryCameraButton"),
  calibrateButton: document.getElementById("calibrateButton"),
  modelStatus: document.getElementById("modelStatus"),
  setupVideo: document.getElementById("setupVideo"),
  setupCameraPlaceholder: document.getElementById("setupCameraPlaceholder"),
  setupEyebrow: document.getElementById("setupEyebrow"),
  setupTitle: document.getElementById("setupTitle"),
  setupDescription: document.getElementById("setupDescription"),
  calibrationWash: document.getElementById("calibrationWash"),
  calibrationStep: document.getElementById("calibrationStep"),
  calibrationCount: document.getElementById("calibrationCount"),
  calibrationMessage: document.getElementById("calibrationMessage"),
  missionEyebrow: document.getElementById("missionEyebrow"),
  emotionName: document.getElementById("emotionName"),
  emotionPrompt: document.getElementById("emotionPrompt"),
  guideIllustration: document.getElementById("guideIllustration"),
  guideEyebrows: document.getElementById("guideEyebrows"),
  guideEyes: document.getElementById("guideEyes"),
  guideNose: document.getElementById("guideNose"),
  guideMouth: document.getElementById("guideMouth"),
  faceGuideItems: [...document.querySelectorAll(".face-guide-item")],
  guideContent: document.getElementById("guideContent"),
  guideToggle: document.getElementById("guideToggle"),
  roundClock: document.getElementById("roundClock"),
  clockProgress: document.getElementById("clockProgress"),
  clockSeconds: document.getElementById("clockSeconds"),
  clockPhase: document.getElementById("clockPhase"),
  clockHelp: document.getElementById("clockHelp"),
  gameCameraStage: document.getElementById("gameCameraStage"),
  gameVideo: document.getElementById("gameVideo"),
  faceCanvas: document.getElementById("faceCanvas"),
  practicePlaceholder: document.getElementById("practicePlaceholder"),
  practiceFace: document.getElementById("practiceFace"),
  cameraBadge: document.getElementById("cameraBadge"),
  phaseBanner: document.getElementById("phaseBanner"),
  phaseBannerLabel: document.getElementById("phaseBannerLabel"),
  phaseBannerText: document.getElementById("phaseBannerText"),
  cameraFeedback: document.querySelector(".camera-feedback"),
  matchMessage: document.getElementById("matchMessage"),
  manualSuccessButton: document.getElementById("manualSuccessButton"),
  skipButton: document.getElementById("skipButton"),
  pauseButton: document.getElementById("pauseButton"),
  gameHomeButton: document.getElementById("gameHomeButton"),
  analysisList: document.getElementById("analysisList"),
  analysisToggle: document.getElementById("analysisToggle"),
  leadingEmotion: document.getElementById("leadingEmotion"),
  leadingNote: document.getElementById("leadingNote"),
  detectVideo: document.getElementById("detectVideo"),
  detectCanvas: document.getElementById("detectCanvas"),
  detectCameraStage: document.getElementById("detectCameraStage"),
  detectCameraBadge: document.getElementById("detectCameraBadge"),
  detectPhaseBanner: document.getElementById("detectPhaseBanner"),
  detectPhaseLabel: document.getElementById("detectPhaseLabel"),
  detectPhaseText: document.getElementById("detectPhaseText"),
  detectClock: document.getElementById("detectClock"),
  detectClockProgress: document.getElementById("detectClockProgress"),
  detectClockSeconds: document.getElementById("detectClockSeconds"),
  detectClockPhase: document.getElementById("detectClockPhase"),
  detectClockHelp: document.getElementById("detectClockHelp"),
  detectStatus: document.getElementById("detectStatus"),
  detectAnalysisList: document.getElementById("detectAnalysisList"),
  detectAnalysisToggle: document.getElementById("detectAnalysisToggle"),
  detectLeadingEmotion: document.getElementById("detectLeadingEmotion"),
  detectLeadingNote: document.getElementById("detectLeadingNote"),
  detectResult: document.getElementById("detectResult"),
  detectResultEyebrow: document.getElementById("detectResultEyebrow"),
  detectResultName: document.getElementById("detectResultName"),
  detectResultDescription: document.getElementById("detectResultDescription"),
  detectResultTop: document.getElementById("detectResultTop"),
  detectPauseButton: document.getElementById("detectPauseButton"),
  detectRetryButton: document.getElementById("detectRetryButton"),
  detectHomeButton: document.getElementById("detectHomeButton"),
  completedCount: document.getElementById("completedCount"),
  matchedCount: document.getElementById("matchedCount"),
  bestEmotion: document.getElementById("bestEmotion"),
  roundRecap: document.getElementById("roundRecap"),
  replayButton: document.getElementById("replayButton"),
  homeButton: document.getElementById("homeButton"),
  roundResult: document.getElementById("roundResult"),
  roundResultEyebrow: document.getElementById("roundResultEyebrow"),
  roundResultTitle: document.getElementById("roundResultTitle"),
  roundResultDescription: document.getElementById("roundResultDescription"),
  roundResultTop: document.getElementById("roundResultTop"),
  roundResultNextButton: document.getElementById("roundResultNextButton"),
  toast: document.getElementById("toast")
};

function bindEvents() {
  ui.cameraStartButton.addEventListener("click", () => startCameraJourney("game"));
  ui.detectStartButton.addEventListener("click", () => startCameraJourney("detect"));
  ui.practiceStartButton.addEventListener("click", startPracticeJourney);
  ui.setupPracticeButton.addEventListener("click", startPracticeJourney);
  ui.retryCameraButton.addEventListener("click", retryCameraJourney);
  ui.calibrateButton.addEventListener("click", calibrateExpressionRange);
  ui.manualSuccessButton.addEventListener("click", completePracticeEmotion);
  ui.skipButton.addEventListener("click", skipEmotion);
  ui.pauseButton.addEventListener("click", togglePause);
  ui.gameHomeButton.addEventListener("click", returnHome);
  ui.detectPauseButton.addEventListener("click", togglePause);
  ui.detectRetryButton.addEventListener("click", startExpressionDetection);
  ui.detectHomeButton.addEventListener("click", returnHome);
  ui.replayButton.addEventListener("click", replayJourney);
  ui.homeButton.addEventListener("click", returnHome);
  ui.soundButton.addEventListener("click", toggleSound);
  ui.analysisToggle.addEventListener("click", toggleAnalysisList);
  ui.detectAnalysisToggle.addEventListener("click", toggleDetectAnalysisList);
  ui.guideToggle.addEventListener("click", toggleGuide);
  ui.roundResultNextButton.addEventListener("click", continueAfterRoundResult);
  document.querySelector(".brand").addEventListener("click", (event) => {
    event.preventDefault();
    returnHome();
  });
  window.addEventListener("resize", () => resizeCanvas());
  window.addEventListener("orientationchange", handleViewportInterruption);
  document.addEventListener("visibilitychange", handleVisibilityChange);
  window.addEventListener("beforeunload", stopCamera);
}

async function startCameraJourney(journey = "game") {
  const requestId = ++state.cameraRequestId;
  state.mode = "camera";
  state.journey = journey;
  resetSession();
  state.baseline = {};
  state.featureScales = {};
  setScreen("setup");
  updateSetupCopy();
  setModelStatus("카메라 사용 권한을 확인하고 있어요.", "loading");
  ui.calibrateButton.disabled = true;
  ui.calibrateButton.classList.remove("hidden");
  ui.retryCameraButton.classList.add("hidden");
  ui.setupCameraPlaceholder.classList.remove("hidden");
  ui.setupCameraPlaceholder.querySelector("strong").textContent = "카메라를 준비하고 있어요";
  setCameraStartPending(true);

  if (!navigator.mediaDevices?.getUserMedia) {
    handleCameraError(new Error("이 브라우저에서는 카메라를 사용할 수 없습니다."));
    return;
  }

  try {
    const stream = await requestCameraStream();
    if (!isCurrentCameraRequest(requestId)) {
      stopMediaStream(stream);
      return;
    }
    state.stream = stream;

    ui.setupVideo.srcObject = state.stream;
    ui.gameVideo.srcObject = state.stream;
    ui.detectVideo.srcObject = state.stream;
    await Promise.all([
      safePlay(ui.setupVideo),
      withTimeout(loadFaceModel(), MODEL_LOAD_TIMEOUT_MS, "얼굴 분석기 연결 시간이 너무 오래 걸리고 있어요.")
    ]);
    if (!isCurrentCameraRequest(requestId)) return;
    ui.setupCameraPlaceholder.classList.add("hidden");
    setModelStatus("얼굴 분석기가 준비됐어요. 화면 가운데를 바라봐 주세요.", "loading");
    startDetectionLoop();
  } catch (error) {
    if (isCurrentCameraRequest(requestId)) handleCameraError(error);
  } finally {
    if (requestId === state.cameraRequestId) setCameraStartPending(false);
  }
}

function retryCameraJourney() {
  const journey = state.journey;
  stopCamera();
  startCameraJourney(journey);
}

function isCurrentCameraRequest(requestId) {
  return requestId === state.cameraRequestId && state.mode === "camera" && state.screen === "setup";
}

function setCameraStartPending(pending) {
  state.cameraStarting = pending;
  ui.cameraStartButton.disabled = pending;
  ui.detectStartButton.disabled = pending;
  ui.retryCameraButton.disabled = pending;
  ui.setupScreen.setAttribute("aria-busy", String(pending));
}

async function requestCameraStream() {
  const preferred = {
    video: {
      facingMode: "user",
      width: { ideal: isMobileViewport() ? 640 : 960 },
      height: { ideal: isMobileViewport() ? 480 : 720 },
      frameRate: isMobileViewport() ? { ideal: 12, max: 15 } : { ideal: 24, max: 30 }
    },
    audio: false
  };
  try {
    return await navigator.mediaDevices.getUserMedia(preferred);
  } catch (error) {
    if (error?.name !== "OverconstrainedError") throw error;
    return navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
  }
}

function startPracticeJourney() {
  state.mode = "practice";
  state.journey = "game";
  resetSession();
  stopCamera();
  startGame();
  showToast("연습 모드에서는 표정을 지은 뒤 직접 다음 단계로 이동해요.");
}

async function loadFaceModel() {
  if (state.faceLandmarker) return state.faceLandmarker;
  if (state.faceModelPromise) return state.faceModelPromise;

  setModelStatus("얼굴 분석기를 불러오고 있어요.", "loading");
  state.faceModelPromise = (async () => {
    const { FilesetResolver, FaceLandmarker } = await import(MEDIAPIPE_MODULE);
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);

    const options = {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    };

    try {
      state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, options);
    } catch (gpuError) {
      state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: { modelAssetPath: FACE_MODEL }
      });
    }
    return state.faceLandmarker;
  })();

  try {
    return await state.faceModelPromise;
  } finally {
    state.faceModelPromise = null;
  }
}

function startDetectionLoop() {
  cancelAnimationFrame(state.animationFrameId);
  state.lastDetectionAt = 0;
  state.lastVideoTime = -1;

  const detect = (now) => {
    if (!state.stream || !state.faceLandmarker) return;

    const video = state.screen === "setup"
      ? ui.setupVideo
      : state.screen === "detect"
        ? ui.detectVideo
        : ui.gameVideo;
    const ready = video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0;
    const newFrame = video.currentTime !== state.lastVideoTime;
    const analysisPaused = document.hidden || state.userPaused ||
      (state.screen === "game" && state.successLocked) ||
      (state.screen === "detect" && state.detectPhase === "result");

    if (!analysisPaused && ready && newFrame && now - state.lastDetectionAt >= state.detectionInterval) {
      state.lastDetectionAt = now;
      state.lastVideoTime = video.currentTime;
      const startedAt = performance.now();
      try {
        const result = state.faceLandmarker.detectForVideo(video, now);
        processDetectionResult(result, now);
        const inferenceTime = performance.now() - startedAt;
        state.detectionInterval = clamp(inferenceTime * 1.8, 72, 140);
      } catch (error) {
        console.warn("얼굴 분석 프레임을 처리하지 못했습니다.", error);
      }
    }

    state.animationFrameId = requestAnimationFrame(detect);
  };

  state.animationFrameId = requestAnimationFrame(detect);
}

function processDetectionResult(result, now) {
  const categories = result.faceBlendshapes?.[0]?.categories;
  const landmarks = result.faceLandmarks?.[0];

  if (!categories) {
    handleFaceLost(now);
    clearActiveCanvas();
    return;
  }

  const faceQuality = assessFaceQuality(landmarks);
  if (!faceQuality.ok) {
    handleFaceQualityIssue(now, faceQuality.message);
    clearActiveCanvas();
    return;
  }

  state.currentBlendshapes = Object.fromEntries(categories.map((category) => [category.categoryName, category.score]));
  state.faceDetected = true;

  if (!state.faceStableSince) state.faceStableSince = now;

  if (state.screen === "setup") {
    ui.calibrateButton.disabled = false;
    setModelStatus("얼굴을 찾았어요. 이제 내 표정의 기준을 맞춰 볼까요?", "ready");
    return;
  }

  if (state.screen === "detect") {
    setDetectionFaceStatus(true);
    drawLandmarks(landmarks, ui.detectVideo, ui.detectCanvas);
    if (state.detectPhase === "result") return;
    analyzeAllEmotions();
    updateDetectAnalysisUI();
    tickDetectionClock(now);
    return;
  }

  if (state.screen !== "game") return;

  setFaceStatus(true);
  drawLandmarks(landmarks, ui.gameVideo, ui.faceCanvas);
  analyzeAllEmotions();
  updateAnalysisUI();
  tickRoundClock(now);
  evaluateTargetMatch(now);
  cycleGuideHighlight(now);
}

function handleFaceLost(now) {
  state.currentBlendshapes = null;
  state.faceDetected = false;
  state.faceStableSince = 0;
  state.lastClockAt = now;
  state.targetHoldStarted = 0;

  if (state.screen === "setup") {
    ui.calibrateButton.disabled = true;
    setModelStatus("얼굴을 찾고 있어요. 화면 가운데를 바라봐 주세요.", "loading");
  }

  if (state.screen === "game" && state.mode === "camera") {
    setFaceStatus(false);
    ui.roundClock.classList.add("is-paused");
    ui.gameCameraStage.classList.add("is-paused");
    ui.clockHelp.textContent = "얼굴이 보이면 시간이 다시 흘러요";
    ui.phaseBannerLabel.textContent = "잠시 멈춤";
    ui.phaseBannerText.textContent = "얼굴을 화면 가운데에 맞춰 주세요";
    ui.matchMessage.textContent = "괜찮아요. 얼굴이 다시 보이면 이어서 시작할게요.";
  }

  if (state.screen === "detect") pauseDetectionForFace(now, "얼굴이 다시 보이면 5초 판독을 이어갈게요.");
}

function handleFaceQualityIssue(now, message) {
  state.faceDetected = false;
  state.faceStableSince = 0;
  state.lastClockAt = now;
  state.detectLastAt = now;
  state.targetHoldStarted = 0;

  if (state.screen === "setup") {
    ui.calibrateButton.disabled = true;
    setModelStatus(message, "loading");
  }

  if (state.screen === "game") {
    setFaceStatus(false);
    ui.roundClock.classList.add("is-paused");
    ui.gameCameraStage.classList.add("is-paused");
    ui.clockHelp.textContent = "얼굴 위치를 맞추면 이어져요";
    ui.phaseBannerLabel.textContent = "위치 확인";
    ui.phaseBannerText.textContent = message;
    ui.matchMessage.textContent = message;
  }

  if (state.screen === "detect") pauseDetectionForFace(now, message);
}

function assessFaceQuality(landmarks) {
  if (!landmarks?.length) return { ok: false, message: "얼굴을 화면 가운데에 맞춰 주세요." };
  const xs = landmarks.map((point) => point.x);
  const ys = landmarks.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = maxX - minX;
  const height = maxY - minY;
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  if (height < 0.3 || width < 0.2) return { ok: false, message: "카메라에 조금 더 가까이 와 주세요." };
  if (height > 0.9) return { ok: false, message: "카메라에서 조금만 멀어져 주세요." };
  if (Math.abs(centerX - 0.5) > 0.18 || Math.abs(centerY - 0.5) > 0.18) {
    return { ok: false, message: "얼굴을 화면 가운데로 옮겨 주세요." };
  }
  return { ok: true, message: "" };
}

async function calibrateExpressionRange() {
  if (!state.currentBlendshapes) {
    showToast("얼굴이 화면 가운데에 보인 뒤 다시 눌러 주세요.");
    return;
  }

  ui.calibrateButton.disabled = true;
  ui.calibrationWash.classList.remove("hidden");
  playTone("start");

  const neutralSamples = await captureCalibrationPhase({
    label: "편안한 얼굴",
    message: "힘을 빼고 정면을 바라봐요."
  });

  const smileSamples = await captureCalibrationPhase({
    label: "활짝 웃는 얼굴",
    message: "양쪽 입꼬리를 가장 크게 올려요."
  });

  ui.calibrationWash.classList.add("hidden");

  if (neutralSamples.length < 4 || smileSamples.length < 4) {
    ui.calibrateButton.disabled = false;
    showToast("얼굴을 잠시 놓쳤어요. 화면 가운데에서 다시 맞춰 볼까요?");
    return;
  }

  state.baseline = averageSamples(neutralSamples);
  const smileAverage = averageSamples(smileSamples);
  const observedSmile = average([
    Math.max(0, (smileAverage.mouthSmileLeft || 0) - (state.baseline.mouthSmileLeft || 0)),
    Math.max(0, (smileAverage.mouthSmileRight || 0) - (state.baseline.mouthSmileRight || 0))
  ]);
  const smileScale = clamp(0.45 / Math.max(observedSmile, 0.12), 0.75, 1.55);
  state.featureScales = {
    mouthSmileLeft: smileScale,
    mouthSmileRight: smileScale,
    cheekSquintLeft: smileScale,
    cheekSquintRight: smileScale
  };

  playTone("ready");
  startSelectedJourney();
}

function updateSetupCopy() {
  const isDetect = state.journey === "detect";
  ui.setupEyebrow.textContent = isDetect ? "5초 표정 읽기 · 카메라 준비" : "카메라 준비";
  ui.setupTitle.innerHTML = isDetect ? "내 표정의<br />기준을 먼저 맞춰요" : "내 표정의<br />기준을 맞춰요";
  ui.setupDescription.textContent = isDetect
    ? "정확한 5초 판독을 위해 편안한 얼굴과 활짝 웃는 얼굴을 차례로 보여 주세요."
    : "정면을 바라보고 편안한 얼굴과 활짝 웃는 얼굴을 차례로 보여 주세요.";
  ui.calibrateButton.textContent = isDetect ? "보정하고 5초 읽기" : "표정 기준 맞추기";
  ui.setupPracticeButton.textContent = isDetect ? "대신 게임 둘러보기" : "연습 모드로 계속";
}

function startSelectedJourney() {
  if (state.journey === "detect") startExpressionDetection();
  else startGame();
}

async function captureCalibrationPhase({ label, message }) {
  const samples = [];
  ui.calibrationStep.textContent = label;
  ui.calibrationMessage.textContent = message;

  for (const count of [3, 2, 1]) {
    ui.calibrationCount.textContent = String(count);
    await delay(620);
    if (state.currentBlendshapes && state.faceDetected) samples.push({ ...state.currentBlendshapes });
  }

  ui.calibrationCount.textContent = "✓";
  for (let index = 0; index < 7; index += 1) {
    await delay(110);
    if (state.currentBlendshapes && state.faceDetected) samples.push({ ...state.currentBlendshapes });
  }
  await delay(230);
  return samples;
}

function startGame() {
  state.currentEmotionIndex = 0;
  state.results = [];
  state.successLocked = false;
  setScreen("game");
  loadEmotion(0);
  if (state.mode === "camera") safePlay(ui.gameVideo);
}

function startExpressionDetection() {
  state.journey = "detect";
  state.mode = "camera";
  state.rawScores = createEmptyScores();
  state.smoothedScores = createEmptyScores();
  state.detectPhase = "waiting";
  state.detectElapsed = 0;
  state.detectLastAt = 0;
  state.detectSamples = [];
  state.detectResult = null;
  state.faceStableSince = 0;
  state.userPaused = false;
  state.autoPaused = false;
  setScreen("detect");
  renderScoreList(ui.detectAnalysisList);
  updateDetectAnalysisUI();
  updateDetectionClockUI("waiting", 0);
  ui.detectResult.classList.add("hidden");
  ui.detectAnalysisList.classList.remove("is-expanded");
  ui.detectAnalysisToggle.setAttribute("aria-expanded", "false");
  ui.detectAnalysisToggle.querySelector("span").textContent = "전체 10개 보기";
  ui.detectRetryButton.classList.add("hidden");
  ui.detectPauseButton.classList.remove("hidden");
  ui.detectCameraStage.classList.remove("is-reading", "is-paused");
  ui.detectPhaseBanner.classList.remove("is-final");
  ui.detectPhaseLabel.textContent = "대기";
  ui.detectPhaseText.textContent = "얼굴을 화면 가운데에 맞춰요";
  ui.detectStatus.textContent = "얼굴이 안정적으로 보이면 5초 판독이 자동으로 시작돼요.";
  setDetectionFaceStatus(false);
  updatePauseButtons();
  safePlay(ui.detectVideo);
}

function tickDetectionClock(now) {
  if (state.detectPhase === "result" || state.userPaused || !state.faceDetected) return;

  ui.detectClock.classList.remove("is-paused");
  ui.detectCameraStage.classList.remove("is-paused");

  if (state.detectPhase === "waiting") {
    if (state.faceStableSince && now - state.faceStableSince >= FACE_STABLE_MS) {
      state.detectPhase = "reading";
      state.detectElapsed = 0;
      state.detectLastAt = now;
      state.detectSamples = [];
      ui.detectCameraStage.classList.add("is-reading");
      ui.detectPhaseLabel.textContent = "판독 중";
      ui.detectPhaseText.textContent = "지금 표정을 5초 동안 보여 주세요";
      ui.detectStatus.textContent = "좋아요. 얼굴을 천천히 유지하며 표정을 읽고 있어요.";
      playTone("start");
    } else {
      updateDetectionClockUI("waiting", 0);
    }
    return;
  }

  if (!state.detectLastAt) state.detectLastAt = now;
  const delta = clamp(now - state.detectLastAt, 0, 180);
  state.detectLastAt = now;
  state.detectElapsed += delta;
  state.detectSamples.push({ ...state.rawScores });
  if (state.detectSamples.length > 80) state.detectSamples.shift();
  updateDetectionClockUI("reading", state.detectElapsed);

  if (state.detectElapsed >= DETECT_DURATION_MS) finishExpressionDetection();
}

function updateDetectionClockUI(phase, elapsed) {
  const ratio = phase === "reading" ? clamp(elapsed / DETECT_DURATION_MS, 0, 1) : 0;
  const seconds = phase === "result" ? "✓" : phase === "waiting" ? "5" : String(Math.max(0, Math.ceil((DETECT_DURATION_MS - elapsed) / 1000)));
  if (ui.detectClockSeconds.textContent !== seconds) ui.detectClockSeconds.textContent = seconds;
  ui.detectClockProgress.style.strokeDashoffset = String(CLOCK_CIRCUMFERENCE * ratio);
  const copy = {
    waiting: ["얼굴 확인", "얼굴이 보이면 시작해요"],
    reading: ["5초 판독", "표정을 읽고 있어요"],
    result: ["판독 완료", "결과를 확인해요"]
  }[phase];
  if (ui.detectClockPhase.textContent !== copy[0]) ui.detectClockPhase.textContent = copy[0];
  if (ui.detectClockHelp.textContent !== copy[1]) ui.detectClockHelp.textContent = copy[1];
}

function finishExpressionDetection() {
  if (state.detectPhase === "result") return;
  state.detectPhase = "result";
  state.detectElapsed = DETECT_DURATION_MS;
  const scores = averageScoreSamples(state.detectSamples.length ? state.detectSamples : [state.rawScores]);
  const ranked = rankScores(scores);
  const primary = ranked[0];
  const secondary = ranked[1];
  const isUnclear = primary.score < 0.32;
  const isMixed = !isUnclear && primary.score - secondary.score < 0.08;
  state.detectResult = {
    detectedId: primary.id,
    detectedName: primary.name,
    secondaryName: isMixed ? secondary.name : "",
    clarity: isUnclear ? "unclear" : isMixed ? "mixed" : "clear",
    topScores: ranked.slice(0, 3)
  };

  ui.detectCameraStage.classList.remove("is-reading", "is-paused");
  ui.detectPhaseBanner.classList.add("is-final");
  ui.detectPhaseLabel.textContent = "완료";
  ui.detectPhaseText.textContent = "5초 동안 보인 얼굴 움직임을 정리했어요";
  ui.detectStatus.textContent = "판독이 끝났어요. 아래 결과와 감정별 수치를 확인해 보세요.";
  ui.detectResultEyebrow.textContent = isUnclear ? "가장 가까운 표정 참고값" : isMixed ? "두 표정이 함께 보여요" : "가장 가깝게 읽힌 표정";
  ui.detectResultName.textContent = isMixed ? `${primary.name} + ${secondary.name}` : primary.name;
  ui.detectResultDescription.textContent = isUnclear
    ? `하나의 표정으로 뚜렷하게 나뉘지는 않았지만 ${primary.name} 표정과 가장 가까웠어요.`
    : isMixed
      ? `${primary.name}과 ${secondary.name}에 가까운 얼굴 움직임이 함께 나타났어요.`
      : `얼굴 움직임이 ${primary.name} 표정과 가장 가까웠어요.`;
  ui.detectResultTop.innerHTML = ranked.slice(0, 3).map((item) => `<span>${item.name} ${Math.round(item.score * 100)}%</span>`).join("");
  ui.detectResult.classList.remove("hidden");
  ui.detectRetryButton.classList.remove("hidden");
  ui.detectPauseButton.classList.add("hidden");
  updateDetectionClockUI("result", DETECT_DURATION_MS);
  playTone("result");
}

function pauseDetectionForFace(now, message) {
  state.detectLastAt = now;
  ui.detectClock.classList.add("is-paused");
  ui.detectCameraStage.classList.add("is-paused");
  setDetectionFaceStatus(false);
  ui.detectPhaseLabel.textContent = "잠시 멈춤";
  ui.detectPhaseText.textContent = message;
  ui.detectStatus.textContent = message;
}

function setDetectionFaceStatus(found) {
  ui.detectCameraBadge.classList.toggle("is-ready", found);
  ui.detectCameraBadge.lastChild.textContent = found ? " 얼굴 준비됨" : " 얼굴 찾는 중";
}

function loadEmotion(index) {
  const emotion = EMOTIONS[index];
  if (!emotion) {
    finishGame();
    return;
  }

  state.currentEmotionIndex = index;
  state.rawScores = createEmptyScores();
  state.smoothedScores = createEmptyScores();
  state.finalSamples = [];
  state.roundPhase = state.mode === "camera" ? "waiting" : "practice";
  state.phaseElapsed = 0;
  state.lastClockAt = 0;
  state.faceStableSince = 0;
  state.targetHoldStarted = 0;
  state.successLocked = false;
  state.userPaused = false;
  state.guidePartIndex = 0;
  state.lastGuideChangeAt = 0;

  ui.missionEyebrow.textContent = emotion.orderLabel;
  ui.emotionName.textContent = emotion.name;
  ui.emotionPrompt.textContent = emotion.prompt;
  ui.guideIllustration.setAttribute("aria-label", `${emotion.name} 표정 예시`);
  ui.guideIllustration.innerHTML = createEmotionIllustration(emotion.id);
  ui.guideEyebrows.textContent = emotion.guide.eyebrows;
  ui.guideEyes.textContent = emotion.guide.eyes;
  ui.guideNose.textContent = emotion.guide.nose;
  ui.guideMouth.textContent = emotion.guide.mouth;
  ui.practiceFace.textContent = practiceSymbol(emotion.id);
  ui.sessionProgressText.textContent = `${String(index + 1).padStart(2, "0")} / ${EMOTIONS.length}`;
  ui.sessionProgressBar.style.width = `${((index + 1) / EMOTIONS.length) * 100}%`;

  renderAnalysisList(emotion.id);
  updateAnalysisUI();

  const isPractice = state.mode === "practice";
  ui.practicePlaceholder.classList.toggle("hidden", !isPractice);
  ui.gameVideo.classList.toggle("hidden", isPractice);
  ui.faceCanvas.classList.toggle("hidden", isPractice);
  ui.cameraBadge.classList.toggle("hidden", isPractice);
  ui.manualSuccessButton.classList.toggle("hidden", !isPractice);
  ui.pauseButton.classList.toggle("hidden", isPractice);
  ui.cameraFeedback.classList.toggle("is-practice", isPractice);
  ui.analysisList.classList.remove("is-expanded");
  ui.analysisToggle.setAttribute("aria-expanded", "false");
  ui.analysisToggle.querySelector("span").textContent = "전체 10개 보기";
  collapseGuideOnMobile();
  ui.gameCameraStage.classList.remove("is-final", "is-paused");
  ui.phaseBanner.classList.remove("is-final");
  ui.roundClock.classList.remove("is-final", "is-paused");
  updatePauseButtons();

  if (isPractice) {
    updateClockUI("practice", 0, 1);
    ui.phaseBannerLabel.textContent = "연습";
    ui.phaseBannerText.textContent = "가이드를 보며 내 방식대로 표현해요";
    ui.matchMessage.textContent = "표정을 지은 뒤 ‘이 표정을 지었어요’를 눌러 주세요.";
    ui.leadingEmotion.textContent = "연습 모드";
    ui.leadingNote.textContent = "카메라를 켜면 10개 표정 분석이 나타나요.";
  } else {
    updateClockUI("waiting", 0, 1);
    ui.phaseBannerLabel.textContent = "대기";
    ui.phaseBannerText.textContent = "얼굴을 화면 가운데에 맞춰요";
    ui.matchMessage.textContent = "얼굴이 안정적으로 보이면 3초 준비가 시작돼요.";
  }

  highlightGuidePart(0);
}

function tickRoundClock(now) {
  if (state.mode !== "camera" || state.successLocked || !state.faceDetected || state.userPaused) return;

  ui.roundClock.classList.remove("is-paused");
  ui.gameCameraStage.classList.remove("is-paused");

  if (state.roundPhase === "waiting") {
    if (state.faceStableSince && now - state.faceStableSince >= FACE_STABLE_MS) {
      transitionRoundPhase("prepare", now);
    } else {
      updateClockUI("waiting", 0, 1);
    }
    return;
  }

  if (!state.lastClockAt) state.lastClockAt = now;
  const delta = clamp(now - state.lastClockAt, 0, 180);
  state.lastClockAt = now;
  state.phaseElapsed += delta;

  const duration = ROUND_TIMES[state.roundPhase];
  updateClockUI(state.roundPhase, state.phaseElapsed, duration);

  if (state.roundPhase === "prepare" && state.phaseElapsed >= ROUND_TIMES.prepare) {
    transitionRoundPhase("challenge", now);
    return;
  }

  if (state.roundPhase === "challenge" && state.phaseElapsed >= ROUND_TIMES.challenge) {
    transitionRoundPhase("final", now);
    return;
  }

  if (state.roundPhase === "final") {
    if (state.phaseElapsed >= FINAL_SAMPLE_FROM_MS) {
      state.finalSamples.push({ ...state.rawScores });
      if (state.finalSamples.length > 28) state.finalSamples.shift();
    }
    if (state.phaseElapsed >= ROUND_TIMES.final) resolveRoundByAnalysis();
  }
}

function transitionRoundPhase(phase, now) {
  state.roundPhase = phase;
  state.phaseElapsed = 0;
  state.lastClockAt = now;
  state.targetHoldStarted = 0;

  if (phase === "prepare") {
    ui.phaseBannerLabel.textContent = "준비";
    ui.phaseBannerText.textContent = "가이드를 보고 표정을 떠올려요";
    ui.matchMessage.textContent = "곧 10초 도전이 시작돼요.";
    playTone("start");
  }

  if (phase === "challenge") {
    ui.phaseBannerLabel.textContent = "도전";
    ui.phaseBannerText.textContent = "목표 표정을 10초 안에 보여 주세요";
    ui.matchMessage.textContent = "좋아요. 표정을 천천히 크게 만들어 보세요.";
    playTone("ready");
  }

  if (phase === "final") {
    state.finalSamples = [];
    ui.roundClock.classList.add("is-final");
    ui.gameCameraStage.classList.add("is-final");
    ui.phaseBanner.classList.add("is-final");
    ui.phaseBannerLabel.textContent = "최종 분석";
    ui.phaseBannerText.textContent = "지금 표정을 차분히 읽고 있어요";
    ui.matchMessage.textContent = "마지막 5초예요. 원하는 표정을 그대로 보여 주세요.";
    playTone("final");
  }

  updateClockUI(phase, 0, ROUND_TIMES[phase] || 1);
}

function updateClockUI(phase, elapsed, duration) {
  const ratio = clamp(elapsed / duration, 0, 1);
  const remaining = phase === "practice" ? "—" : phase === "waiting" ? "·" : Math.max(0, Math.ceil((duration - elapsed) / 1000));
  if (ui.clockSeconds.textContent !== String(remaining)) ui.clockSeconds.textContent = String(remaining);
  ui.clockProgress.style.strokeDashoffset = String(CLOCK_CIRCUMFERENCE * ratio);

  const copy = {
    waiting: ["얼굴 확인", "얼굴이 보이면 시간이 시작돼요"],
    prepare: ["준비", "표정을 떠올리는 시간"],
    challenge: ["10초 도전", "목표 표정을 보여 주세요"],
    final: ["최종 분석", "마지막 표정을 읽고 있어요"],
    practice: ["연습", "시간 제한 없이 둘러봐요"]
  }[phase];

  if (ui.clockPhase.textContent !== copy[0]) ui.clockPhase.textContent = copy[0];
  if (ui.clockHelp.textContent !== copy[1]) ui.clockHelp.textContent = copy[1];
}

function analyzeAllEmotions() {
  if (!state.currentBlendshapes) return;
  for (const emotion of EMOTIONS) {
    const rawScore = clamp(
      emotion.customScore
        ? emotion.customScore(state.currentBlendshapes, state.baseline)
        : scoreExpression(emotion, state.currentBlendshapes, state.baseline),
      0,
      1
    );
    state.rawScores[emotion.id] = rawScore;
    const previous = state.smoothedScores[emotion.id] || 0;
    state.smoothedScores[emotion.id] = previous === 0 ? rawScore : previous * 0.72 + rawScore * 0.28;
  }
}

function scoreExpression(emotion, blendshapes, baseline) {
  const totalWeight = emotion.features.reduce((sum, feature) => sum + feature.weight, 0);
  let score = emotion.features.reduce((sum, feature) => {
    const value = featureDelta(feature.names, blendshapes, baseline);
    return sum + clamp(value / feature.target, 0, 1) * feature.weight;
  }, 0) / totalWeight;

  for (const penalty of emotion.penalties || []) {
    const value = featureDelta(penalty.names, blendshapes, baseline);
    score -= clamp(value / penalty.target, 0, 1) * penalty.weight;
  }

  return score;
}

function scoreCalmExpression(blendshapes, baseline) {
  const activityFeatures = [
    "browDownLeft", "browDownRight", "browInnerUp", "eyeWideLeft", "eyeWideRight",
    "eyeSquintLeft", "eyeSquintRight", "jawOpen", "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthPressLeft", "mouthPressRight", "noseSneerLeft", "noseSneerRight"
  ];
  const deltas = activityFeatures
    .map((name) => Math.abs((blendshapes[name] || 0) - (baseline[name] || 0)) * (state.featureScales[name] || 1))
    .sort((a, b) => b - a)
    .slice(0, 6);
  const activity = average(deltas);
  return clamp(1 - activity / 0.23, 0, 1);
}

function featureDelta(names, blendshapes, baseline) {
  return average(names.map((name) => Math.max(0, (blendshapes[name] || 0) - (baseline[name] || 0)) * (state.featureScales[name] || 1)));
}

function evaluateTargetMatch(now) {
  if (state.roundPhase !== "challenge" || state.successLocked) {
    state.targetHoldStarted = 0;
    return;
  }

  const target = EMOTIONS[state.currentEmotionIndex];
  const score = state.smoothedScores[target.id] || 0;
  const ranked = rankScores(state.smoothedScores);
  const targetIsLeader = ranked[0]?.id === target.id;
  const leadGap = (ranked[0]?.score || 0) - (ranked[1]?.score || 0);

  if (score >= target.threshold && targetIsLeader && leadGap >= 0.06) {
    if (!state.targetHoldStarted) state.targetHoldStarted = now;
    const heldFor = now - state.targetHoldStarted;
    ui.matchMessage.textContent = heldFor < SUCCESS_HOLD_MS
      ? "목표 표정과 가까워요. 그대로 잠깐 유지해요."
      : "목표 표정을 찾았어요!";
    if (heldFor >= SUCCESS_HOLD_MS) completeMatchedEmotion();
  } else {
    state.targetHoldStarted = 0;
    if (score >= target.threshold * 0.72) {
      ui.matchMessage.textContent = "거의 닮았어요. 얼굴 부위 가이드를 하나씩 따라 해 보세요.";
    }
  }
}

function renderAnalysisList(targetId) {
  renderScoreList(ui.analysisList, targetId);
}

function renderScoreList(container, targetId = "") {
  container.innerHTML = EMOTIONS.map((emotion) => `
    <div class="emotion-score${emotion.id === targetId ? " is-target" : ""}" data-emotion="${emotion.id}">
      <span class="emotion-score__name">${emotion.name}</span>
      <span class="emotion-score__track"><i class="emotion-score__fill"></i></span>
      <span class="emotion-score__value">0%</span>
    </div>`).join("");
}

function updateAnalysisUI() {
  const ranked = rankScores(state.smoothedScores);
  const leader = ranked[0];
  const second = ranked[1];
  const rankById = new Map(ranked.map((emotion, index) => [emotion.id, index]));

  for (const emotion of EMOTIONS) {
    const row = ui.analysisList.querySelector(`[data-emotion="${emotion.id}"]`);
    if (!row) continue;
    const rank = rankById.get(emotion.id) ?? EMOTIONS.length;
    const percent = Math.round((state.smoothedScores[emotion.id] || 0) * 100);
    row.querySelector(".emotion-score__fill").style.width = `${percent}%`;
    row.querySelector(".emotion-score__value").textContent = `${percent}%`;
    row.style.setProperty("--rank", String(rank + 1));
    row.classList.toggle("is-top-three", rank < 3);
    row.classList.toggle("is-leading", Boolean(leader && leader.id === emotion.id && leader.score > 0.03));
  }

  if (!leader || leader.score < 0.04) {
    ui.leadingEmotion.textContent = state.mode === "practice" ? "연습 모드" : "분석 준비 중";
    ui.leadingNote.textContent = state.mode === "practice" ? "카메라를 켜면 실시간 분석이 나타나요." : "표정을 지으면 결과가 나타나요.";
    return;
  }

  ui.leadingEmotion.textContent = `${leader.name} ${Math.round(leader.score * 100)}%`;
  if (second && leader.score - second.score < 0.08) {
    ui.leadingNote.textContent = `${second.name} 표정도 함께 나타나고 있어요.`;
  } else {
    ui.leadingNote.textContent = "현재 얼굴 움직임과 가장 가까운 표정이에요.";
  }
}

function updateDetectAnalysisUI() {
  const ranked = updateScoreList(ui.detectAnalysisList, state.smoothedScores);
  const leader = ranked[0];
  const second = ranked[1];
  if (!leader || leader.score < 0.04) {
    ui.detectLeadingEmotion.textContent = "분석 준비 중";
    ui.detectLeadingNote.textContent = "표정을 지으면 수치가 나타나요.";
    return;
  }
  ui.detectLeadingEmotion.textContent = `${leader.name} ${Math.round(leader.score * 100)}%`;
  ui.detectLeadingNote.textContent = second && leader.score - second.score < 0.08
    ? `${second.name} 표정도 함께 나타나고 있어요.`
    : "현재 얼굴 움직임과 가장 가까운 표정이에요.";
}

function updateScoreList(container, scores) {
  const ranked = rankScores(scores);
  const leader = ranked[0];
  const rankById = new Map(ranked.map((emotion, index) => [emotion.id, index]));
  for (const emotion of EMOTIONS) {
    const row = container.querySelector(`[data-emotion="${emotion.id}"]`);
    if (!row) continue;
    const rank = rankById.get(emotion.id) ?? EMOTIONS.length;
    const percent = Math.round((scores[emotion.id] || 0) * 100);
    row.querySelector(".emotion-score__fill").style.width = `${percent}%`;
    row.querySelector(".emotion-score__value").textContent = `${percent}%`;
    row.style.setProperty("--rank", String(rank + 1));
    row.classList.toggle("is-top-three", rank < 3);
    row.classList.toggle("is-leading", Boolean(leader && leader.id === emotion.id && leader.score > 0.03));
  }
  return ranked;
}

function toggleAnalysisList() {
  const expanded = !ui.analysisList.classList.contains("is-expanded");
  ui.analysisList.classList.toggle("is-expanded", expanded);
  ui.analysisToggle.setAttribute("aria-expanded", String(expanded));
  ui.analysisToggle.querySelector("span").textContent = expanded ? "상위 3개만 보기" : "전체 10개 보기";
  if (expanded && isMobileViewport()) setGuideExpanded(false);
}

function toggleDetectAnalysisList() {
  const expanded = !ui.detectAnalysisList.classList.contains("is-expanded");
  ui.detectAnalysisList.classList.toggle("is-expanded", expanded);
  ui.detectAnalysisToggle.setAttribute("aria-expanded", String(expanded));
  ui.detectAnalysisToggle.querySelector("span").textContent = expanded ? "상위 3개만 보기" : "전체 10개 보기";
}

function toggleGuide() {
  const expanded = !ui.guideContent.classList.contains("is-expanded");
  setGuideExpanded(expanded);
  if (expanded && isMobileViewport()) {
    ui.analysisList.classList.remove("is-expanded");
    ui.analysisToggle.setAttribute("aria-expanded", "false");
    ui.analysisToggle.querySelector("span").textContent = "전체 10개 보기";
  }
}

function collapseGuideOnMobile() {
  setGuideExpanded(!isMobileViewport());
}

function setGuideExpanded(expanded) {
  ui.guideContent.classList.toggle("is-expanded", expanded);
  ui.guideToggle.setAttribute("aria-expanded", String(expanded));
  ui.guideToggle.querySelector("span").textContent = expanded ? "가이드 접기" : "가이드 보기";
}

function isMobileViewport() {
  return window.matchMedia("(max-width: 720px)").matches;
}

function togglePause() {
  if (state.mode !== "camera") return;
  state.userPaused = !state.userPaused;
  state.autoPaused = false;
  const now = performance.now();
  state.lastClockAt = now;
  state.detectLastAt = now;
  updatePauseButtons();

  if (state.screen === "game") {
    ui.roundClock.classList.toggle("is-paused", state.userPaused);
    ui.gameCameraStage.classList.toggle("is-paused", state.userPaused);
    if (state.userPaused) {
      ui.phaseBannerLabel.textContent = "일시정지";
      ui.phaseBannerText.textContent = "준비되면 다시 시작해요";
      ui.matchMessage.textContent = "시간을 잠시 멈췄어요.";
    } else {
      ui.phaseBannerLabel.textContent = state.roundPhase === "final" ? "최종 분석" : state.roundPhase === "challenge" ? "도전" : "준비";
      ui.phaseBannerText.textContent = state.roundPhase === "final" ? "지금 표정을 차분히 읽고 있어요" : state.roundPhase === "challenge" ? "목표 표정을 10초 안에 보여 주세요" : "가이드를 보고 표정을 떠올려요";
      ui.matchMessage.textContent = "좋아요. 멈춘 곳부터 이어서 시작할게요.";
    }
  }

  if (state.screen === "detect") {
    ui.detectClock.classList.toggle("is-paused", state.userPaused);
    ui.detectCameraStage.classList.toggle("is-paused", state.userPaused);
    ui.detectPhaseLabel.textContent = state.userPaused ? "일시정지" : state.detectPhase === "reading" ? "판독 중" : "대기";
    ui.detectPhaseText.textContent = state.userPaused ? "준비되면 다시 시작해요" : state.detectPhase === "reading" ? "지금 표정을 5초 동안 보여 주세요" : "얼굴을 화면 가운데에 맞춰요";
    ui.detectStatus.textContent = state.userPaused ? "5초 판독을 잠시 멈췄어요." : "멈춘 곳부터 표정 판독을 이어갈게요.";
  }
}

function handleVisibilityChange() {
  if (document.hidden) {
    pauseForDeviceInterruption("다른 화면으로 이동해 자동으로 멈췄어요.");
    setCameraTracksEnabled(false);
    return;
  }
  setCameraTracksEnabled(true);
  state.faceStableSince = 0;
  if (state.screen === "setup" && state.mode === "camera") {
    setModelStatus("카메라를 다시 확인하고 있어요. 화면 가운데를 바라봐 주세요.", "loading");
  }
}

function setCameraTracksEnabled(enabled) {
  if (!state.stream) return;
  for (const track of state.stream.getVideoTracks()) track.enabled = enabled;
}

function handleViewportInterruption() {
  pauseForDeviceInterruption("화면 방향이 바뀌어 자동으로 멈췄어요.");
  window.setTimeout(() => resizeCanvas(), 180);
}

function pauseForDeviceInterruption(message) {
  if (state.mode !== "camera" || !["setup", "game", "detect"].includes(state.screen)) return;
  if (state.screen === "setup") {
    state.faceStableSince = 0;
    setModelStatus(message + " 화면을 다시 확인해 주세요.", "loading");
    return;
  }
  if (state.userPaused) return;
  state.userPaused = true;
  state.autoPaused = true;
  state.lastClockAt = performance.now();
  state.detectLastAt = performance.now();
  updatePauseButtons();
  if (state.screen === "game") {
    ui.roundClock.classList.add("is-paused");
    ui.gameCameraStage.classList.add("is-paused");
    ui.phaseBannerLabel.textContent = "자동 멈춤";
    ui.phaseBannerText.textContent = "계속하기를 누르면 이어져요";
    ui.matchMessage.textContent = message;
  }
  if (state.screen === "detect") {
    ui.detectClock.classList.add("is-paused");
    ui.detectCameraStage.classList.add("is-paused");
    ui.detectPhaseLabel.textContent = "자동 멈춤";
    ui.detectPhaseText.textContent = "계속하기를 누르면 이어져요";
    ui.detectStatus.textContent = message;
  }
}

function updatePauseButtons() {
  const label = state.userPaused ? "계속하기" : "일시정지";
  for (const button of [ui.pauseButton, ui.detectPauseButton]) {
    button.textContent = label;
    button.setAttribute("aria-pressed", String(state.userPaused));
  }
}

function resolveRoundByAnalysis() {
  if (state.successLocked) return;
  const finalScores = averageScoreSamples(state.finalSamples.length ? state.finalSamples : [state.rawScores]);
  const ranked = rankScores(finalScores);
  const primary = ranked[0];
  const secondary = ranked[1];
  const target = EMOTIONS[state.currentEmotionIndex];
  const isUnclear = primary.score < 0.32;
  const isMixed = !isUnclear && primary.score - secondary.score < 0.08;

  const result = {
    targetId: target.id,
    targetName: target.name,
    detectedId: primary.id,
    detectedName: primary.name,
    secondaryName: isMixed ? secondary.name : "",
    matched: primary.id === target.id && !isUnclear && !isMixed,
    method: "analysis",
    clarity: isUnclear ? "unclear" : isMixed ? "mixed" : "clear",
    topScores: ranked.slice(0, 3),
    skipped: false
  };

  finishRound(result);
}

function completeMatchedEmotion() {
  if (state.successLocked) return;
  const target = EMOTIONS[state.currentEmotionIndex];
  const ranked = rankScores(state.smoothedScores);
  finishRound({
    targetId: target.id,
    targetName: target.name,
    detectedId: target.id,
    detectedName: target.name,
    secondaryName: "",
    matched: true,
    method: "matched",
    clarity: "clear",
    topScores: ranked.slice(0, 3),
    skipped: false
  });
}

function completePracticeEmotion() {
  if (state.successLocked) return;
  const target = EMOTIONS[state.currentEmotionIndex];
  finishRound({
    targetId: target.id,
    targetName: target.name,
    detectedId: target.id,
    detectedName: target.name,
    secondaryName: "",
    matched: true,
    method: "practice",
    clarity: "practice",
    topScores: [{ id: target.id, name: target.name, score: 1 }],
    skipped: false
  });
}

function skipEmotion() {
  if (state.successLocked) return;
  const target = EMOTIONS[state.currentEmotionIndex];
  finishRound({
    targetId: target.id,
    targetName: target.name,
    detectedId: "skip",
    detectedName: "건너뜀",
    secondaryName: "",
    matched: false,
    method: "skip",
    clarity: "skip",
    topScores: [],
    skipped: true
  }, true);
}

function finishRound(result, skipOverlay = false) {
  if (state.successLocked) return;
  state.successLocked = true;
  state.results.push(result);
  const sessionVersion = state.sessionVersion;

  if (skipOverlay) {
    showToast(`${result.targetName} 표정은 건너뛰었어요. 다음에 다시 표현해 볼 수 있어요.`);
    window.setTimeout(() => {
      if (sessionVersion === state.sessionVersion && state.screen === "game") {
        loadEmotion(state.currentEmotionIndex + 1);
      }
    }, 280);
    return;
  }

  const copy = createRoundResultCopy(result);
  ui.roundResultEyebrow.textContent = copy.eyebrow;
  ui.roundResultTitle.textContent = copy.title;
  ui.roundResultTitle.classList.toggle("is-long", copy.title.length > 7);
  ui.roundResultDescription.textContent = copy.description;
  ui.roundResultTop.innerHTML = result.topScores.map((item) => `<span>${item.name} ${Math.round(item.score * 100)}%</span>`).join("");
  ui.roundResultNextButton.textContent = state.currentEmotionIndex === EMOTIONS.length - 1 ? "탐색 결과 보기" : "다음 표정으로";
  state.pendingRoundVersion = sessionVersion;
  setRoundResultOpen(true);
  playTone(result.matched ? "success" : "result");
}

function continueAfterRoundResult() {
  const canContinue = state.pendingRoundVersion === state.sessionVersion && state.screen === "game";
  state.pendingRoundVersion = 0;
  setRoundResultOpen(false, false);
  if (!canContinue) return;
  loadEmotion(state.currentEmotionIndex + 1);
  window.requestAnimationFrame(() => ui.emotionName.focus({ preventScroll: true }));
}

function setRoundResultOpen(open, restoreFocus = true) {
  const main = document.querySelector("main");
  const topbar = document.querySelector(".topbar");
  ui.roundResult.classList.toggle("hidden", !open);
  main.inert = open;
  topbar.inert = open;

  if (open) {
    hideToast();
    state.roundResultReturnFocus = document.activeElement;
    window.requestAnimationFrame(() => ui.roundResultNextButton.focus());
    return;
  }

  if (restoreFocus && state.roundResultReturnFocus?.isConnected) {
    state.roundResultReturnFocus.focus({ preventScroll: true });
  }
  state.roundResultReturnFocus = null;
}

function createRoundResultCopy(result) {
  if (result.method === "matched" || result.method === "practice") {
    return {
      eyebrow: "목표 표정을 찾았어요",
      title: result.targetName,
      description: "목표 표정과 가까운 얼굴 움직임이 나타났어요."
    };
  }

  if (result.clarity === "unclear") {
    return {
      eyebrow: "표정에는 여러 모습이 있어요",
      title: result.detectedName,
      description: `뚜렷하게 하나로 나뉘지는 않았지만 ${result.detectedName} 표정과 가장 가까웠어요.`
    };
  }

  if (result.clarity === "mixed") {
    return {
      eyebrow: "두 가지 표정이 함께 보여요",
      title: `${result.detectedName} + ${result.secondaryName}`,
      description: "한 얼굴에 여러 감정의 움직임이 함께 나타났어요."
    };
  }

  return {
    eyebrow: "가장 가깝게 읽힌 표정",
    title: result.detectedName,
    description: `목표는 ${result.targetName}, 카메라는 ${result.detectedName} 표정과 가장 가깝게 읽었어요.`
  };
}

function finishGame() {
  state.successLocked = false;
  if (state.mode === "camera") stopCamera();
  setScreen("result");

  const completed = state.results.filter((result) => !result.skipped);
  const matched = completed.filter((result) => result.matched);
  const detectedCounts = completed.reduce((counts, result) => {
    counts[result.detectedName] = (counts[result.detectedName] || 0) + 1;
    return counts;
  }, {});
  const mostDetected = Object.entries(detectedCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "여러 표정";

  ui.completedCount.textContent = String(completed.length);
  ui.matchedCount.textContent = String(matched.length);
  ui.bestEmotion.textContent = mostDetected;
  ui.roundRecap.innerHTML = state.results.map((result, index) => `
    <li>
      <span>${String(index + 1).padStart(2, "0")}</span>
      <strong>${result.targetName}</strong>
      <em>→ ${result.detectedName}${result.secondaryName ? ` + ${result.secondaryName}` : ""}</em>
    </li>`).join("");
  playTone("finish");
}

function replayJourney() {
  resetSession();
  if (state.mode === "camera") {
    startCameraJourney("game");
  } else {
    startPracticeJourney();
  }
}

function returnHome() {
  resetSession();
  stopCamera();
  setScreen("welcome");
}

function resetSession() {
  state.sessionVersion += 1;
  state.pendingRoundVersion = 0;
  setRoundResultOpen(false, false);
  state.currentEmotionIndex = 0;
  state.results = [];
  state.rawScores = createEmptyScores();
  state.smoothedScores = createEmptyScores();
  state.finalSamples = [];
  state.roundPhase = "idle";
  state.phaseElapsed = 0;
  state.lastClockAt = 0;
  state.faceStableSince = 0;
  state.targetHoldStarted = 0;
  state.successLocked = false;
  state.userPaused = false;
  state.autoPaused = false;
  state.detectPhase = "idle";
  state.detectElapsed = 0;
  state.detectLastAt = 0;
  state.detectSamples = [];
  state.detectResult = null;
  ui.retryCameraButton.classList.add("hidden");
}

function setScreen(name) {
  state.screen = name;
  ui.welcomeScreen.classList.toggle("hidden", name !== "welcome");
  ui.setupScreen.classList.toggle("hidden", name !== "setup");
  ui.gameScreen.classList.toggle("hidden", name !== "game");
  ui.detectScreen.classList.toggle("hidden", name !== "detect");
  ui.resultScreen.classList.toggle("hidden", name !== "result");
  ui.sessionProgress.classList.toggle("hidden", name !== "game");
  window.scrollTo({ top: 0, behavior: isMobileViewport() ? "auto" : "smooth" });
  if (name === "game") resizeCanvas(ui.gameVideo, ui.faceCanvas);
  if (name === "detect") resizeCanvas(ui.detectVideo, ui.detectCanvas);
}

function setModelStatus(message, type) {
  ui.modelStatus.querySelector("span:last-child").textContent = message;
  ui.modelStatus.classList.toggle("is-ready", type === "ready");
  ui.modelStatus.classList.toggle("is-error", type === "error");
}

function setFaceStatus(found) {
  ui.cameraBadge.classList.toggle("is-ready", found);
  ui.cameraBadge.lastChild.textContent = found ? " 얼굴 준비됨" : " 얼굴 찾는 중";
}

function cycleGuideHighlight(now) {
  if (!state.lastGuideChangeAt || now - state.lastGuideChangeAt >= 1500) {
    state.lastGuideChangeAt = now;
    highlightGuidePart(state.guidePartIndex);
    state.guidePartIndex = (state.guidePartIndex + 1) % ui.faceGuideItems.length;
  }
}

function highlightGuidePart(index) {
  ui.faceGuideItems.forEach((item, itemIndex) => item.classList.toggle("is-active", itemIndex === index));
}

function rankScores(scores) {
  return EMOTIONS.map((emotion) => ({
    id: emotion.id,
    name: emotion.name,
    score: clamp(scores[emotion.id] || 0, 0, 1)
  })).sort((a, b) => b.score - a.score);
}

function averageScoreSamples(samples) {
  const output = createEmptyScores();
  for (const emotion of EMOTIONS) {
    output[emotion.id] = average(samples.map((sample) => sample[emotion.id] || 0));
  }
  return output;
}

function createEmptyScores() {
  return Object.fromEntries(EMOTIONS.map((emotion) => [emotion.id, 0]));
}

function handleCameraError(error) {
  console.warn("카메라 시작 오류", error);
  let message = "카메라를 시작하지 못했어요. 연습 모드로 계속할 수 있어요.";
  let placeholder = "카메라를 열지 못했어요";
  if (error?.name === "NotAllowedError") message = "카메라 권한이 꺼져 있어요. 브라우저 설정에서 허용하거나 연습 모드를 이용해 주세요.";
  if (error?.name === "NotFoundError") message = "연결된 카메라를 찾지 못했어요. 연습 모드로 계속할 수 있어요.";
  if (error?.name === "NotReadableError") message = "다른 앱에서 카메라를 사용하고 있어요. 다른 앱을 닫은 뒤 다시 시도해 주세요.";
  if (error?.name === "OverconstrainedError") message = "이 기기의 카메라 설정을 맞추지 못했어요. 브라우저를 새로 고친 뒤 다시 시도해 주세요.";
  if (/fetch|dynamically imported module|network/i.test(error?.message || "")) {
    message = "얼굴 분석기를 불러오지 못했어요. 인터넷 연결을 확인한 뒤 다시 시도해 주세요.";
    placeholder = "얼굴 분석기를 불러오지 못했어요";
  }
  if (error?.name === "TimeoutError") {
    message = "얼굴 분석기 연결이 지연되고 있어요. 인터넷 연결을 확인한 뒤 다시 시도해 주세요.";
    placeholder = "연결 시간이 오래 걸리고 있어요";
  }
  stopCamera();
  setModelStatus(message, "error");
  ui.calibrateButton.classList.add("hidden");
  ui.setupCameraPlaceholder.classList.remove("hidden");
  ui.setupCameraPlaceholder.querySelector("strong").textContent = placeholder;
  ui.retryCameraButton.classList.remove("hidden");
}

function stopCamera() {
  state.cameraRequestId += 1;
  setCameraStartPending(false);
  cancelAnimationFrame(state.animationFrameId);
  state.animationFrameId = 0;
  if (state.stream) {
    stopMediaStream(state.stream);
    state.stream = null;
  }
  ui.setupVideo.srcObject = null;
  ui.gameVideo.srcObject = null;
  ui.detectVideo.srcObject = null;
  state.currentBlendshapes = null;
  state.faceDetected = false;
  state.lastVideoTime = -1;
}

function stopMediaStream(stream) {
  stream?.getTracks?.().forEach((track) => track.stop());
}

function drawLandmarks(landmarks, video, canvas) {
  if (!landmarks?.length || state.mode !== "camera") return;
  resizeCanvas(video, canvas);
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "rgba(255, 228, 160, 0.38)";
  for (let index = 0; index < landmarks.length; index += 4) {
    const point = landmarks[index];
    context.beginPath();
    context.arc(point.x * canvas.width, point.y * canvas.height, 1.15, 0, Math.PI * 2);
    context.fill();
  }
}

function resizeCanvas(video = state.screen === "detect" ? ui.detectVideo : ui.gameVideo, canvas = state.screen === "detect" ? ui.detectCanvas : ui.faceCanvas) {
  if (!video?.videoWidth || !canvas) return;
  if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
  if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
}

function clearActiveCanvas() {
  const canvas = state.screen === "detect" ? ui.detectCanvas : ui.faceCanvas;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function toggleSound() {
  state.soundEnabled = !state.soundEnabled;
  ui.soundButton.setAttribute("aria-pressed", String(state.soundEnabled));
  ui.soundButton.setAttribute("aria-label", state.soundEnabled ? "소리 켜짐" : "소리 꺼짐");
  ui.soundLabel.textContent = state.soundEnabled ? "소리 켬" : "소리 끔";
  ui.soundButton.querySelector("span:first-child").textContent = state.soundEnabled ? "♪" : "×";
  if (state.soundEnabled) playTone("ready");
}

function playTone(kind) {
  if (!state.soundEnabled) return;
  try {
    state.audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
    const context = state.audioContext;
    const notes = {
      start: [392],
      ready: [523, 659],
      final: [440, 554],
      success: [523, 659, 784],
      result: [440, 554, 659],
      finish: [392, 523, 659, 784]
    }[kind] || [523];
    const startedAt = context.currentTime;
    notes.forEach((frequency, index) => {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = "sine";
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0.0001, startedAt + index * 0.1);
      gain.gain.exponentialRampToValueAtTime(0.075, startedAt + index * 0.1 + 0.018);
      gain.gain.exponentialRampToValueAtTime(0.0001, startedAt + index * 0.1 + 0.22);
      oscillator.connect(gain).connect(context.destination);
      oscillator.start(startedAt + index * 0.1);
      oscillator.stop(startedAt + index * 0.1 + 0.24);
    });
  } catch (error) {
    console.warn("효과음을 재생하지 못했습니다.", error);
  }
}

function showToast(message) {
  hideToast();
  ui.toast.textContent = message;
  ui.toast.classList.remove("hidden");
  state.toastTimer = window.setTimeout(() => ui.toast.classList.add("hidden"), 4200);
}

function hideToast() {
  window.clearTimeout(state.toastTimer);
  state.toastTimer = 0;
  ui.toast.classList.add("hidden");
}

function averageSamples(samples) {
  const names = new Set(samples.flatMap((sample) => Object.keys(sample)));
  return Object.fromEntries([...names].map((name) => [name, average(samples.map((sample) => sample[name] || 0))]));
}

function practiceSymbol(id) {
  return {
    joy: "☺", sad: "︵", anger: "ಠ", surprise: "○", fear: "◉",
    disgust: "⌇", worry: "﹏", shy: "⌣", excited: "★", calm: "˘"
  }[id] || "☺";
}

function createEmotionIllustration(id) {
  const expressions = {
    joy: {
      brows: '<path d="M80 119c15-10 31-10 44-1M196 118c14-9 30-8 43 2"/>',
      eyes: '<path d="M83 147c12 13 29 13 40 0M197 147c11 13 28 13 39 0"/>',
      mouth: '<path class="mouth-fill" d="M113 188c23 35 70 35 94-1-25 14-67 14-94 1Z"/>',
      extra: '<ellipse class="cheek" cx="83" cy="180" rx="23" ry="11"/><ellipse class="cheek" cx="239" cy="180" rx="23" ry="11"/>'
    },
    sad: {
      brows: '<path d="M80 123c15-12 30-15 44-8M196 115c15-7 30-4 44 8"/>',
      eyes: '<path d="M83 150c12 5 27 5 39-1M198 149c11 6 26 6 38 0"/>',
      mouth: '<path d="M123 207c20-22 55-22 75 0"/>',
      extra: '<path class="tear" d="M231 157c-12 18-12 25 0 31 12-6 12-13 0-31Z"/>'
    },
    anger: {
      brows: '<path d="M78 110c17 2 32 8 46 18M242 110c-18 2-33 9-46 18"/>',
      eyes: '<path d="M84 151c13-5 28-4 39 2M198 153c12-6 27-6 39-1"/>',
      mouth: '<path d="M121 203c23-13 53-13 78 0"/>',
      extra: '<path class="accent-line" d="M247 84l13-13M254 97l20-3"/>'
    },
    surprise: {
      brows: '<path d="M80 106c14-13 31-15 45-4M195 102c15-11 32-9 44 4"/>',
      eyes: '<ellipse cx="103" cy="148" rx="11" ry="17"/><ellipse cx="217" cy="148" rx="11" ry="17"/>',
      mouth: '<ellipse class="mouth-fill" cx="160" cy="203" rx="24" ry="31"/>',
      extra: '<path class="accent-line" d="M265 103l18-7M262 118l22 4"/>'
    },
    fear: {
      brows: '<path d="M80 111c17-13 32-14 45-4M195 107c14-10 30-9 45 4"/>',
      eyes: '<ellipse cx="103" cy="149" rx="12" ry="17"/><ellipse cx="217" cy="149" rx="12" ry="17"/>',
      mouth: '<path class="mouth-fill" d="M126 216c12-36 56-36 68 0-19-12-49-12-68 0Z"/>',
      extra: '<path class="sweat" d="M248 105c-13 19-13 28 0 34 13-6 13-15 0-34Z"/>'
    },
    disgust: {
      brows: '<path d="M80 119c15-7 30-5 43 4M197 122c14-7 29-5 42 3"/>',
      eyes: '<path d="M84 151c12 6 26 6 38 0M198 151c11 6 25 6 37 0"/>',
      mouth: '<path d="M121 199c18 9 27-9 43 0 15 8 23-8 39-2"/>',
      extra: '<path class="nose-line" d="M148 168c8 8 17 8 24 0M141 174c6 8 13 11 20 10"/>'
    },
    worry: {
      brows: '<path d="M78 123c18-15 34-17 47-8M195 115c14-9 30-7 47 8"/>',
      eyes: '<ellipse cx="103" cy="151" rx="9" ry="12"/><ellipse cx="217" cy="151" rx="9" ry="12"/>',
      mouth: '<path d="M129 204c20-13 42-13 62 0"/>',
      extra: '<path class="accent-line" d="M264 126c12 7 17 16 17 28"/>'
    },
    shy: {
      brows: '<path d="M82 121c13-7 27-7 40 0M199 121c13-7 27-7 39 0"/>',
      eyes: '<path d="M85 153c11 8 25 8 36 0M200 153c10 8 24 8 35 0"/>',
      mouth: '<path d="M134 193c16 15 36 15 52 0"/>',
      extra: '<ellipse class="cheek cheek--strong" cx="85" cy="181" rx="28" ry="13"/><ellipse class="cheek cheek--strong" cx="237" cy="181" rx="28" ry="13"/>'
    },
    excited: {
      brows: '<path d="M80 108c14-11 30-11 44-2M196 106c15-9 31-8 43 2"/>',
      eyes: '<path class="star-eye" d="M102 132l5 11 12 1-9 8 3 12-11-6-11 6 3-12-9-8 12-1Z"/><path class="star-eye" d="M218 132l5 11 12 1-9 8 3 12-11-6-11 6 3-12-9-8 12-1Z"/>',
      mouth: '<path class="mouth-fill" d="M115 188c25 45 67 45 91 0-28 16-64 16-91 0Z"/>',
      extra: '<path class="accent-line" d="M263 91l12-15M269 106l19-3"/>'
    },
    calm: {
      brows: '<path d="M82 120c14-7 28-7 41 0M198 120c13-7 27-7 39 0"/>',
      eyes: '<path d="M84 151c12 11 27 11 38 0M199 151c11 11 25 11 36 0"/>',
      mouth: '<path d="M133 193c17 14 37 14 54 0"/>',
      extra: '<path class="accent-line" d="M261 101c9-7 18-7 27 0M268 114c7-5 14-5 21 0"/>'
    }
  };

  const expression = expressions[id] || expressions.joy;
  return `
    <svg viewBox="0 0 320 300" role="img" aria-hidden="true">
      <style>
        .blob{fill:#f0c866;stroke:#4d4037;stroke-width:3;stroke-linejoin:round}.ear{fill:#efb584;stroke:#4d4037;stroke-width:4}.face{fill:#efb584;stroke:#4d4037;stroke-width:4}.hair{fill:#745747;stroke:#4d4037;stroke-width:4;stroke-linejoin:round}.feature{fill:none;stroke:#4d4037;stroke-width:6;stroke-linecap:round;stroke-linejoin:round}.feature ellipse{fill:#4d4037;stroke:none}.mouth-fill{fill:#fffaf0;stroke:#4d4037}.cheek{fill:#ef7b66;opacity:.35;stroke:none}.cheek--strong{opacity:.55}.tear,.sweat{fill:#9ccbd0;stroke:#4d4037;stroke-width:2}.accent-line,.nose-line{fill:none;stroke:#d95f4d;stroke-width:4;stroke-linecap:round}.star-eye{fill:#4d4037;stroke:none}.shirt{fill:#9ccbd0;stroke:#4d4037;stroke-width:4}
      </style>
      <path class="blob" d="M34 141C22 73 75 22 148 27c75-17 142 31 139 102 21 65-26 129-96 135-67 25-140-7-151-69-17-16-20-36-6-54Z"/>
      <path class="shirt" d="M91 282c10-40 37-61 69-61 34 0 61 22 70 61Z"/>
      <ellipse class="ear" cx="65" cy="155" rx="24" ry="31"/><ellipse class="ear" cx="255" cy="155" rx="24" ry="31"/>
      <path class="face" d="M64 124c0-64 39-96 96-96s96 34 96 99v45c0 67-41 101-96 101s-96-36-96-101Z"/>
      <path class="hair" d="M65 127C54 64 93 28 154 28c63 0 102 38 101 101-23-35-48-49-78-50-34 25-70 38-112 48Z"/>
      <g class="feature">${expression.brows}${expression.eyes}${expression.mouth}</g>
      ${expression.extra}
    </svg>`;
}

async function safePlay(video) {
  try {
    const playPromise = video.play();
    if (playPromise) await Promise.race([playPromise, delay(2000)]);
  } catch {
    // 재생 준비 상태는 분석 루프에서 다시 확인한다.
  }
}

function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function withTimeout(promise, milliseconds, message) {
  let timeoutId;
  const timeout = new Promise((_, reject) => {
    timeoutId = window.setTimeout(() => {
      const error = new Error(message);
      error.name = "TimeoutError";
      reject(error);
    }, milliseconds);
  });
  return Promise.race([promise, timeout]).finally(() => window.clearTimeout(timeoutId));
}

function average(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

window.render_game_to_text = () => JSON.stringify({
  screen: state.screen,
  mode: state.mode,
  journey: state.journey,
  paused: state.userPaused,
  autoPaused: state.autoPaused,
  game: state.screen === "game" ? {
    emotion: EMOTIONS[state.currentEmotionIndex]?.name || null,
    round: state.currentEmotionIndex + 1,
    phase: state.roundPhase,
    elapsedMs: Math.round(state.phaseElapsed),
    leader: rankScores(state.smoothedScores)[0]
  } : null,
  detection: state.screen === "detect" ? {
    phase: state.detectPhase,
    elapsedMs: Math.round(state.detectElapsed),
    leader: rankScores(state.smoothedScores)[0],
    result: state.detectResult
  } : null
});

window.advanceTime = (milliseconds) => {
  let remaining = Math.max(0, Number(milliseconds) || 0);
  let now = performance.now();
  while (remaining > 0) {
    const step = Math.min(100, remaining);
    now += step;
    if (state.screen === "detect") tickDetectionClock(now);
    if (state.screen === "game") tickRoundClock(now);
    remaining -= step;
  }
  return window.render_game_to_text();
};

bindEvents();
setScreen("welcome");
