// 단계 상태와 카메라 표시를 자세 편집기의 모델 조작에서 분리한다.
export function createWorkflowPanel({server, post, onChange, onTargets, notify}){
  const $ = id => document.getElementById(id);
  const terminal = new Set(["ARRIVED", "FAILED", "STOPPED"]);
  let stageState = null, connected = false, externalBusy = true, pending = null;
  let mode = "color", cameraActive = false, cameraPending = false, cameraTimer = null;
  let disposed = false, polling = false, frameURL = null, observedRun = null;

  function stageBusy(){ return pending !== null || stageState?.state === "MOVING"; }

  function renderStage(){
    const state = stageState?.state ?? "IDLE";
    const labels = {IDLE:"1단계 대기", MOVING:"1단계 이동 중", ARRIVED:"1단계 완료", FAILED:"실행 실패", STOPPED:"중지됨"};
    $("workflow-state").textContent = labels[state] ?? "상태 확인 중";
    $("workflow-state").dataset.state = state;
    $("stage1-start").disabled = !connected || externalBusy || stageBusy() || stageState === null;
    $("stage1-start").dataset.complete = String(state === "ARRIVED");
    $("stage1-action").textContent = pending === "start" ? "명령 전송 중…" : state === "MOVING" ? "이동 중…" : state === "ARRIVED" ? "도착 완료 · 다시 실행" : "실제 모터 이동";
    $("workflow-stop").disabled = !connected || pending === "stop" || !stageBusy();
    $("workflow-stop").textContent = pending === "stop" ? "중지 중…" : "단계 중지";
  }

  function acceptStage(status){
    if (!status) return;
    if (status.run_id === stageState?.run_id && terminal.has(stageState?.state) && status.state === "MOVING") return;
    stageState = status;
    $("workflow-message").textContent = status.state === "ARRIVED"
      ? "1단계 도착 완료. 정면 보정은 2단계에서 진행합니다."
      : status.message;
    if (status.stop_error) $("workflow-message").textContent += ` 정지 확인 실패: ${status.stop_error}`;
    if (status.run_id && status.run_id !== observedRun && Object.keys(status.targets_deg ?? {}).length){
      observedRun = status.run_id;
      onTargets(status.targets_deg);
    }
    renderStage();
    onChange();
  }

  $("stage1-start").addEventListener("click", async () => {
    if ($("stage1-start").disabled) return;
    pending = "start";
    renderStage(); onChange();
    try { acceptStage(await post("/api/stage1/start", {})); }
    catch (error){ $("workflow-message").textContent = error.message; notify(error.message); }
    finally { if (pending === "start") pending = null; renderStage(); onChange(); }
  });

  async function stopStage(){
    if (!stageBusy()) return;
    pending = "stop";
    renderStage(); onChange();
    try { acceptStage(await post("/api/stage1/stop", {})); }
    catch (error){ $("workflow-message").textContent = error.message; notify(error.message); }
    finally { pending = null; renderStage(); onChange(); }
  }
  $("workflow-stop").addEventListener("click", () => {
    if (!$("workflow-stop").disabled) stopStage();
  });

  function clearFrame(){
    $("camera-image").hidden = true;
    $("camera-image").removeAttribute("src");
    if (frameURL) URL.revokeObjectURL(frameURL);
    frameURL = null;
    $("camera-placeholder").hidden = false;
  }

  function renderCamera(state, message){
    const labels = {STOPPED:"꺼짐", STARTING:"연결 중", LIVE:"실시간", STALLED:"수신 대기", ERROR:"연결 확인"};
    $("camera-state").textContent = labels[state] ?? "연결 확인";
    $("camera-state").dataset.state = state;
    $("camera-toggle").disabled = cameraPending;
    $("camera-toggle").textContent = cameraPending ? "처리 중…" : cameraActive ? "카메라 끄기" : "카메라 연결";
    $("camera-caption").textContent = state === "LIVE"
      ? mode === "depth" ? "깊이 100–700 mm · 검정: 측정 없음" : "실물 카메라 · RGB 영상"
      : message;
    if (state !== "LIVE"){
      clearFrame();
      $("camera-placeholder-text").textContent = state === "STARTING" ? "카메라 연결 중…" : state === "STOPPED" ? "카메라 화면이 꺼져 있습니다" : "새 영상을 기다립니다";
    }
  }

  async function cameraStatus(){
    const response = await fetch(server + "/api/camera", {cache:"no-store", signal:AbortSignal.timeout(4000)});
    if (!response.ok) throw new Error(response.status === 404 ? "카메라 화면을 사용하려면 서버를 재시작해 주세요." : "카메라 상태를 읽지 못했습니다.");
    return response.json();
  }

  async function pollCamera(){
    if (disposed || !cameraActive || polling) return;
    polling = true;
    let delay = 100;
    const requestedMode = mode;
    try {
      const response = await fetch(server + "/api/camera/frame/" + requestedMode, {cache:"no-store", signal:AbortSignal.timeout(4000)});
      if (!response.ok){
        const status = await cameraStatus();
        if (["ERROR", "STOPPED"].includes(status.state)) cameraActive = false;
        renderCamera(status.state, status.message);
        delay = 400;
      } else {
        const blob = await response.blob();
        if (disposed || !cameraActive || requestedMode !== mode) return;
        const nextURL = URL.createObjectURL(blob);
        const previous = frameURL;
        frameURL = nextURL;
        $("camera-image").src = nextURL;
        $("camera-image").hidden = false;
        $("camera-placeholder").hidden = true;
        if (previous) URL.revokeObjectURL(previous);
        renderCamera("LIVE", "");
      }
    } catch (error){
      renderCamera("ERROR", error.message);
      delay = 1500;
    } finally {
      polling = false;
      if (!disposed && cameraActive) cameraTimer = setTimeout(pollCamera, delay);
    }
  }

  async function startCamera(){
    cameraPending = true;
    renderCamera("STARTING", "카메라를 연결하고 있습니다.");
    try {
      const status = await post("/api/camera/start", {});
      cameraActive = !["ERROR", "STOPPED"].includes(status.state);
      renderCamera(status.state, status.message);
    } catch (error){
      cameraActive = false;
      renderCamera("ERROR", error.message === "Not Found" ? "카메라 화면을 사용하려면 서버를 재시작해 주세요." : error.message);
    }
    finally {
      cameraPending = false;
      $("camera-toggle").disabled = false;
      $("camera-toggle").textContent = cameraActive ? "카메라 끄기" : "카메라 연결";
      if (cameraActive) pollCamera();
    }
  }

  $("camera-toggle").addEventListener("click", async () => {
    if (cameraPending) return;
    if (!cameraActive){ await startCamera(); return; }
    cameraPending = true;
    cameraActive = false;
    clearTimeout(cameraTimer);
    renderCamera("STOPPED", "카메라 표시를 종료하고 있습니다.");
    try { const status = await post("/api/camera/stop", {}); renderCamera(status.state, status.message); }
    catch (error){ renderCamera("ERROR", error.message); }
    finally { cameraPending = false; $("camera-toggle").disabled = false; $("camera-toggle").textContent = "카메라 연결"; }
  });

  document.querySelectorAll("[data-camera-mode]").forEach(button => button.addEventListener("click", () => {
    if (mode === button.dataset.cameraMode) return;
    mode = button.dataset.cameraMode;
    document.querySelectorAll("[data-camera-mode]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
    $("camera-mode-tag").textContent = mode === "color" ? "RGB" : "DEPTH";
    $("camera-image").alt = mode === "color" ? "실시간 RGB 카메라 영상" : "실시간 깊이 카메라 영상";
    clearFrame();
    clearTimeout(cameraTimer);
    pollCamera();
  }));

  // 실시간 표시 시작은 모터 실행과 분리하며 페이지를 여는 것만으로 단계 이동하지 않는다.
  startCamera();
  fetch(server + "/api/stage1", {cache:"no-store", signal:AbortSignal.timeout(4000)})
    .then(response => { if (!response.ok) throw new Error("단계 버튼을 사용하려면 서버를 재시작해 주세요."); return response.json(); })
    .then(acceptStage).catch(error => { $("workflow-message").textContent = error.message; });

  addEventListener("pagehide", () => {
    disposed = true;
    clearTimeout(cameraTimer);
    clearFrame();
  }, {once:true});

  return {
    get busy(){ return stageBusy(); },
    stop: stopStage,
    setContext(context){ connected = context.connected; externalBusy = context.busy; renderStage(); },
    observe(payload){
      if (payload.stage1) acceptStage(payload.stage1);
      if (stageState?.state === "MOVING"){
        const arrived = payload.joints.filter(state => state.arrived_now && state.command_id === stageState.command_ids[state.name]).length;
        $("workflow-message").textContent = `시작 자세로 이동 중 · 관절 도착 ${arrived}/7`;
      }
    },
  };
}
