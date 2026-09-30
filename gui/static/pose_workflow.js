// 단계 상태와 카메라 표시를 자세 편집기의 모델 조작에서 분리해요.
import {createStageSequence} from "./pose_sequence.js?v=20260930-both-grips-eight";

export function createWorkflowPanel({server, post, onChange, onTargets, notify, releaseBody, lockRight, stopAction, setTimer, clearTimer}){
  const $ = id => document.getElementById(id);
  const terminal = new Set(["ARRIVED", "ALIGNED", "REACHED", "FAILED", "STOPPED"]);
  let stageState = null, alignmentState = null, liftState = null, exitState = null, heightState = null, insertionState = null, rearState = null, frontState = null, selectedStage = 1;
  let connected = false, externalBusy = true, pending = null;
  let mode = "color", cameraActive = false, cameraPending = false, cameraTimer = null;
  let disposed = false, polling = false, frameURL = null, observedRun = null, observedStep = null, observedLift = null, observedExit = null, observedHeight = null, observedInsertion = null, observedRear = null, observedFront = null;
  const sequence = createStageSequence({
    startStage: stage => pressStageButton(stage, true),
    async readStage(stage){
      const response = await fetch(`${server}/api/stage${stage}`, {cache:"no-store"});
      if (!response.ok) throw new Error(`${stage}단계 상태를 읽지 못했어요.`);
      const status = await response.json();
      acceptWorkflowStage(stage, status);
      renderWorkflow();
      return status;
    },
    async stopStage(stage){
      const status = await post(`/api/stage${stage}/stop`, {});
      acceptWorkflowStage(stage, status);
      renderWorkflow();
      return status;
    },
    releaseBody,
    lockRight,
    stopAction,
    setTimer,
    clearTimer,
    onChange: renderWorkflow,
  });

  function stageBusy(){ return sequence.active || pending !== null || stageState?.state === "MOVING" || alignmentState?.active === true || liftState?.active === true || exitState?.active === true || heightState?.active === true || insertionState?.active === true || rearState?.active === true || frontState?.active === true; }
  function shownState(){ return selectedStage === 8 ? frontState : selectedStage === 7 ? rearState : selectedStage === 6 ? insertionState : selectedStage === 5 ? heightState : selectedStage === 4 ? exitState : selectedStage === 3 ? liftState : selectedStage === 2 ? alignmentState : stageState; }
  function acceptWorkflowStage(stage, status){
    [acceptStage, acceptAlignment, acceptLift, acceptExit, acceptHeight, acceptInsertion, acceptRear, acceptFront][stage - 1](status);
  }
  function stageRequest(stage){
    return stage >= 7 ? {distance_mm:Number($(`stage${stage}-distance`).value)} : {};
  }
  async function pressStageButton(stage, automatic = false){
    if (!automatic) sequence.reset();
    const request = `start${stage}`;
    pending = request; selectedStage = stage;
    renderStage(); onChange();
    try {
      const status = await post(`/api/stage${stage}/start`, stageRequest(stage));
      acceptWorkflowStage(stage, status);
      renderWorkflow();
      return status;
    }
    catch (error){
      $("workflow-message").textContent = error.message; notify(error.message);
      if (automatic) throw error;
    }
    finally { if (pending === request) pending = null; renderStage(); onChange(); }
  }

  function renderStage(){
    const shown = shownState();
    const state = shown?.state ?? "IDLE";
    const rearDone = rearState?.phase === "close_sent" ? "복귀 완료 · L 닫기 명령 전송" : rearState?.side_return_completed ? "복귀 완료 · 열린 상태" : "당김 완료 · 열린 상태";
    const labels = {IDLE:`${selectedStage}단계 대기`, MOVING:`${selectedStage}단계 이동 중`, OBSERVING:"빔 관측 중", ARRIVED:"1단계 완료", ALIGNED:"정면 보정 완료", REACHED:selectedStage === 8 ? "앞발 잠금각 도착" : selectedStage === 7 ? rearDone : selectedStage === 6 ? "삽입 위치 확인" : selectedStage === 5 ? "목표 높이 도착" : selectedStage === 4 ? "외측 이동 완료" : "첫 상승 완료", FAILED:"실행 실패", STOPPED:"중지됨"};
    $("workflow-state").textContent = labels[state] ?? "상태 확인 중";
    $("workflow-state").dataset.state = state;
    $("sequence-start").disabled = !connected || externalBusy || stageBusy() || stageState === null;
    $("sequence-start").dataset.complete = String(sequence.status.state === "COMPLETE");
    $("sequence-action").textContent = sequence.status.message;
    $("stage1-start").disabled = !connected || externalBusy || stageBusy() || stageState === null;
    $("stage1-start").dataset.complete = String(stageState?.state === "ARRIVED");
    $("stage1-action").textContent = pending === "start1" ? "명령 전송 중…" : stageState?.state === "MOVING" ? "이동 중…" : stageState?.state === "ARRIVED" ? "도착 완료 · 다시 실행" : "실제 모터 이동";
    $("stage2-start").disabled = !connected || externalBusy || stageBusy() || alignmentState === null;
    $("stage2-start").dataset.complete = String(alignmentState?.state === "ALIGNED");
    $("stage2-action").textContent = alignmentState?.active ? "관측·보정 중…" : alignmentState?.state === "ALIGNED" ? "정면 확인 완료" : "관측 후 작은 보정";
    $("stage3-start").disabled = !connected || externalBusy || stageBusy() || liftState === null;
    $("stage3-start").dataset.complete = String(liftState?.state === "REACHED");
    $("stage3-action").textContent = liftState?.active ? "관측·상승 중…" : liftState?.state === "REACHED" ? "목표 간격 도착" : "간격 25 mm까지";
    $("stage4-start").disabled = !connected || externalBusy || stageBusy() || exitState === null;
    $("stage4-start").dataset.complete = String(exitState?.state === "REACHED");
    $("stage4-action").textContent = exitState?.active ? "관측·횡이동 중…" : exitState?.state === "REACHED" ? "고정턱 여유 확보" : "옆 여유 10 mm까지";
    $("stage5-start").disabled = !connected || externalBusy || stageBusy() || heightState === null;
    $("stage5-start").dataset.complete = String(heightState?.state === "REACHED");
    $("stage5-action").textContent = heightState?.active ? "높이·정면 보정 중…" : heightState?.state === "REACHED" ? "목표 높이 도착" : "삽입 높이 맞추기";
    $("stage6-start").disabled = !connected || externalBusy || stageBusy() || insertionState === null;
    $("stage6-start").dataset.complete = String(insertionState?.state === "REACHED");
    $("stage6-action").textContent = insertionState?.active ? "관측·삽입 중…" : insertionState?.state === "REACHED" ? "삽입 위치 확인" : "4단계 출발 횡위치로";
    const rearReady = typeof rearState?.side_return_completed === "boolean";
    $("stage7-start").disabled = !connected || externalBusy || stageBusy() || !rearReady;
    $("stage7-start").dataset.complete = String(Boolean(rearState?.state === "REACHED" && rearState.side_return_completed));
    const rearPhase = {opening:"개방 중…", side_exit:"옆으로 빼는 중…", pulling:"연속 당김 중…", pull_arrival:"당김 도착 확인 중…", return_planning:"횡복귀 계산 중…", side_return:"빔 쪽으로 넣는 중…", side_return_arrival:"복귀 도착 확인 중…", closing:"L 닫기 명령 전송 중…", holding:"현재 자세 유지 중…"};
    $("stage7-action").textContent = rearState && !rearReady ? "서버 재시작 필요" : rearState?.active ? rearPhase[rearState.phase] ?? "경로 계산 중…" : rearState?.state === "REACHED" ? rearDone : "개방·당김·복귀·L 닫기";
    $("stage7-distance").disabled = stageBusy();
    $("stage8-start").disabled = !connected || externalBusy || stageBusy() || frontState === null;
    $("stage8-start").dataset.complete = String(frontState?.state === "REACHED");
    const frontPhase = {planning:"경로 확인 중…", holding:"현재 자세 유지 중…", opening:"앞발 개방 중…", advancing:"직진·높이 보정 중…", closing:"앞발 잠금 중…"};
    $("stage8-action").textContent = frontState?.active ? frontPhase[frontState.phase] ?? "관측 중…" : frontState?.state === "REACHED" ? "직진·잠금각 도착" : "수동 지지 전환 후 시작";
    $("stage8-distance").disabled = stageBusy();
    $("workflow-stop").disabled = !connected || pending === "stop" || sequence.status.state === "STOPPING" || !stageBusy();
    $("workflow-stop").textContent = pending === "stop" || sequence.status.state === "STOPPING" ? "중지 중…" : sequence.active ? "자동 진행 중지" : "단계 중지";
  }

  function acceptStage(status){
    if (!status) return;
    if (status.run_id === stageState?.run_id && terminal.has(stageState?.state) && status.state === "MOVING") return;
    stageState = status;
    if (status.state === "MOVING") selectedStage = 1;
    if (status.run_id && status.run_id !== observedRun && Object.keys(status.targets_deg ?? {}).length){
      observedRun = status.run_id;
      onTargets(status.targets_deg);
    }
  }

  function acceptAlignment(status){
    if (!status) return;
    if (status.run_id === alignmentState?.run_id && terminal.has(alignmentState?.state) && ["OBSERVING", "MOVING"].includes(status.state)) return;
    const newRun = status.run_id && status.run_id !== alignmentState?.run_id;
    alignmentState = status;
    if (status.active || newRun) selectedStage = 2;
    const step = `${status.run_id}:${status.step}`;
    if (status.step > 0 && step !== observedStep && Object.keys(status.targets_deg ?? {}).length){
      observedStep = step;
      onTargets(status.targets_deg);
    }
  }

  function renderWorkflow(){
    if (sequence.active) selectedStage = sequence.status.stage;
    const status = shownState();
    if (status){
      let message = status.message;
      if (selectedStage === 2 && Number.isFinite(status.tilt_deg)) message += ` · ${status.tilt_deg.toFixed(2)}° / 목표 ${status.tolerance_deg}° 이내 · ${status.step}회 보정`;
      if (selectedStage === 3 && Number.isFinite(status.gap_mm)) message += ` · 간격 ${status.gap_mm.toFixed(1)} mm / 목표 ${status.goal_gap_mm} mm · ${status.step}회 이동`;
      if (selectedStage === 4 && Number.isFinite(status.clearance_mm)){
        const gap = status.clearance_mm < 0 ? `빔과 겹침 ${(-status.clearance_mm).toFixed(1)} mm` : `옆 여유 ${status.clearance_mm.toFixed(1)} mm`;
        message += ` · ${gap} / 목표 ${status.goal_clearance_mm} mm · ${status.step}회 이동`;
        if (status.observation?.single_edge) message += " · 한쪽 모서리 추적";
      }
      if (selectedStage === 5 && Number.isFinite(status.remaining_mm)){
        if (Number.isFinite(status.depth_mm) && Number.isFinite(status.goal_depth_mm)){
          message += ` · 빔 거리 ${status.depth_mm.toFixed(2)} mm / 현재 자세 목표 ${status.goal_depth_mm.toFixed(2)} mm`;
        }
        message += ` · 높이 오차 ${status.remaining_mm.toFixed(2)} mm`;
        if (Number.isFinite(status.lower_clearance_mm)) message += ` · 추정 몸체 여유 ${status.lower_clearance_mm.toFixed(1)} mm`;
        if (Number.isFinite(status.tilt_deg)) message += ` · 정면 오차 ${status.tilt_deg.toFixed(2)}°`;
        if (Number.isFinite(status.side_clearance_mm)) message += ` · 옆 ${status.side_clearance_mm.toFixed(1)} mm`;
        message += ` · ${status.step}회 이동`;
      }
      if (selectedStage === 6 && Number.isFinite(status.remaining_mm)){
        message += ` · 복귀 잔여 ${status.remaining_mm.toFixed(1)} mm`;
        if (Number.isFinite(status.height_error_mm)){
          message += ` · 높이 오차 ${status.height_error_mm.toFixed(2)} mm`;
          if (Number.isFinite(status.height_tolerance_mm)) message += ` / 완료 ±${status.height_tolerance_mm.toFixed(1)} mm`;
        }
        if (Number.isFinite(status.tilt_deg)) message += ` · 정면 오차 ${status.tilt_deg.toFixed(2)}°`;
        message += ` · ${status.step}회 보정`;
        if (status.observation?.single_edge) message += " · 한쪽 모서리 추적";
      }
      if (selectedStage === 7){
        if (Number.isFinite(status.side_exit_deg)) message += ` · 옆 빼기 ${status.side_exit_deg}° ${status.side_exit_completed ? "완료" : "예정"}`;
        message += ` · 경로 진행 ${Number(status.planned_progress_mm ?? 0).toFixed(1)} / ${status.distance_mm} mm`;
        if (Number.isFinite(status.side_return_distance_mm)) message += ` · 횡복귀 ${Number(status.planned_return_mm ?? 0).toFixed(1)} / ${status.side_return_distance_mm.toFixed(1)} mm`;
        message += " · 이동량은 관절 기준 추정";
      }
      if (selectedStage === 8){
        message += ` · 전진 ${Number(status.progress_mm ?? 0).toFixed(1)} / ${status.distance_mm} mm (관절 기준)`;
        if (Number.isFinite(status.height_error_mm)) message += ` · 높이 오차 ${status.height_error_mm.toFixed(1)} mm`;
        if (Number.isFinite(status.height_tolerance_mm)) message += ` / 완료 ±${status.height_tolerance_mm.toFixed(1)} mm`;
        if (Number.isFinite(status.lateral_error_mm)) message += ` · 직진 횡오차 ${status.lateral_error_mm.toFixed(1)} mm (관절 기준)`;
      }
      if (status.stop_error) message += ` 정지 확인 실패: ${status.stop_error}`;
      $("workflow-message").textContent = message;
    }
    if (sequence.active) $("workflow-message").textContent = sequence.status.action ? sequence.status.message : `자동 진행 · ${sequence.status.step}/${sequence.status.total} · ${$("workflow-message").textContent}`;
    else if (["COMPLETE", "FAILED", "STOPPED"].includes(sequence.status.state)) $("workflow-message").textContent = sequence.status.message;
    renderStage(); onChange();
  }

  $("sequence-start").addEventListener("click", () => {
    if (!$("sequence-start").disabled) sequence.start();
  });

  for (let stage = 1; stage <= 8; stage++){
    $(`stage${stage}-start`).addEventListener("click", () => {
      if ($(`stage${stage}-start`).disabled) return;
      if (stage >= 7 && !$(`stage${stage}-distance`).reportValidity()) return;
      return pressStageButton(stage);
    });
  }

  function acceptLift(status){
    if (!status) return;
    if (status.run_id === liftState?.run_id && terminal.has(liftState?.state) && ["OBSERVING", "MOVING"].includes(status.state)) return;
    const newRun = status.run_id && status.run_id !== liftState?.run_id;
    liftState = status;
    if (status.active || newRun) selectedStage = 3;
    const step = `${status.run_id}:${status.step}`;
    if (status.step > 0 && step !== observedLift && Object.keys(status.targets_deg ?? {}).length){
      observedLift = step;
      onTargets(status.targets_deg);
    }
  }

  function acceptExit(status){
    if (!status) return;
    if (status.run_id === exitState?.run_id && terminal.has(exitState?.state) && ["OBSERVING", "MOVING"].includes(status.state)) return;
    const newRun = status.run_id && status.run_id !== exitState?.run_id;
    exitState = status;
    if (status.active || newRun) selectedStage = 4;
    const step = `${status.run_id}:${status.step}`;
    if (status.step > 0 && step !== observedExit && Object.keys(status.targets_deg ?? {}).length){
      observedExit = step;
      onTargets(status.targets_deg);
    }
  }

  function acceptHeight(status){
    if (!status) return;
    if (status.run_id === heightState?.run_id && terminal.has(heightState?.state) && ["OBSERVING", "MOVING"].includes(status.state)) return;
    const newRun = status.run_id && status.run_id !== heightState?.run_id;
    heightState = status;
    if (status.active || newRun) selectedStage = 5;
    const step = `${status.run_id}:${status.step}`;
    if (status.step > 0 && step !== observedHeight && Object.keys(status.targets_deg ?? {}).length){
      observedHeight = step;
      onTargets(status.targets_deg);
    }
  }

  function acceptInsertion(status){
    if (!status) return;
    if (status.run_id === insertionState?.run_id && terminal.has(insertionState?.state) && ["OBSERVING", "MOVING"].includes(status.state)) return;
    const newRun = status.run_id && status.run_id !== insertionState?.run_id;
    insertionState = status;
    if (status.active || newRun) selectedStage = 6;
    const step = `${status.run_id}:${status.step}`;
    if (status.step > 0 && step !== observedInsertion && Object.keys(status.targets_deg ?? {}).length){
      observedInsertion = step;
      onTargets(status.targets_deg);
    }
  }

  function acceptRear(status){
    if (!status) return;
    if (status.run_id === rearState?.run_id && terminal.has(rearState?.state) && ["OBSERVING", "MOVING"].includes(status.state)) return;
    const newRun = status.run_id && status.run_id !== rearState?.run_id;
    rearState = status;
    if (status.active || newRun) selectedStage = 7;
    const step = `${status.run_id}:${status.step}:${status.path}`;
    if (status.step > 0 && step !== observedRear && Object.keys(status.targets_deg ?? {}).length){
      observedRear = step;
      onTargets(status.targets_deg, status.fixed_pose);
    }
  }

  function acceptFront(status){
    if (!status) return;
    if (status.run_id === frontState?.run_id && terminal.has(frontState?.state) && ["OBSERVING", "MOVING"].includes(status.state)) return;
    const newRun = status.run_id && status.run_id !== frontState?.run_id;
    frontState = status;
    if (status.active || newRun) selectedStage = 8;
    const step = `${status.run_id}:${status.step}`;
    if (status.step > 0 && step !== observedFront && Object.keys(status.targets_deg ?? {}).length){
      observedFront = step;
      onTargets(status.targets_deg);
    }
  }

  async function stopStage(){
    if (sequence.active){ await sequence.stop(); return; }
    if (!stageBusy()) return;
    const stage = pending === "start8" || frontState?.active ? 8 : pending === "start7" || rearState?.active ? 7 : pending === "start6" || insertionState?.active ? 6 : pending === "start5" || heightState?.active ? 5 : pending === "start4" || exitState?.active ? 4 : pending === "start3" || liftState?.active ? 3 : pending === "start2" || alignmentState?.active ? 2 : 1;
    pending = "stop";
    renderStage(); onChange();
    try {
      const status = await post(`/api/stage${stage}/stop`, {});
      acceptWorkflowStage(stage, status);
      renderWorkflow();
    }
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
    .then(status => { acceptStage(status); renderWorkflow(); }).catch(error => { $("workflow-message").textContent = error.message; });
  fetch(server + "/api/stage2", {cache:"no-store", signal:AbortSignal.timeout(4000)})
    .then(response => { if (!response.ok) throw new Error("2단계 기능을 사용하려면 서버를 재시작해 주세요."); return response.json(); })
    .then(status => { acceptAlignment(status); renderWorkflow(); }).catch(error => { $("workflow-message").textContent = error.message; });
  fetch(server + "/api/stage3", {cache:"no-store", signal:AbortSignal.timeout(4000)})
    .then(response => { if (!response.ok) throw new Error("3단계 기능을 사용하려면 서버를 재시작해 주세요."); return response.json(); })
    .then(status => { acceptLift(status); renderWorkflow(); }).catch(error => { $("workflow-message").textContent = error.message; });
  fetch(server + "/api/stage4", {cache:"no-store", signal:AbortSignal.timeout(4000)})
    .then(response => { if (!response.ok) throw new Error("4단계 기능을 사용하려면 서버를 재시작해 주세요."); return response.json(); })
    .then(status => { acceptExit(status); renderWorkflow(); }).catch(error => { $("workflow-message").textContent = error.message; });
  fetch(server + "/api/stage5", {cache:"no-store", signal:AbortSignal.timeout(4000)})
    .then(response => { if (!response.ok) throw new Error("5단계 기능을 사용하려면 서버를 재시작해 주세요."); return response.json(); })
    .then(status => { acceptHeight(status); renderWorkflow(); }).catch(error => { $("workflow-message").textContent = error.message; });
  fetch(server + "/api/stage6", {cache:"no-store", signal:AbortSignal.timeout(4000)})
    .then(response => { if (!response.ok) throw new Error("6단계 기능을 사용하려면 서버를 재시작해 주세요."); return response.json(); })
    .then(status => { acceptInsertion(status); renderWorkflow(); }).catch(error => { $("workflow-message").textContent = error.message; });

  fetch(server + "/api/stage7", {cache:"no-store", signal:AbortSignal.timeout(4000)})
    .then(response => { if (!response.ok) throw new Error("7단계 기능을 사용하려면 서버를 재시작해 주세요."); return response.json(); })
    .then(status => { acceptRear(status); renderWorkflow(); }).catch(error => { $("workflow-message").textContent = error.message; });

  fetch(server + "/api/stage8", {cache:"no-store", signal:AbortSignal.timeout(4000)})
    .then(response => { if (!response.ok) throw new Error("8단계 기능을 사용하려면 서버를 재시작해 주세요."); return response.json(); })
    .then(status => { acceptFront(status); renderWorkflow(); }).catch(error => { $("workflow-message").textContent = error.message; });

  addEventListener("pagehide", () => {
    disposed = true;
    if (sequence.active) sequence.stop("페이지를 닫아 자동 진행을 중지했어요.");
    clearTimeout(cameraTimer);
    clearFrame();
  }, {once:true});

  return {
    get busy(){ return stageBusy(); },
    stop: stopStage,
    setContext(context){
      const disconnected = connected && !context.connected;
      connected = context.connected;
      externalBusy = context.busy;
      if (disconnected && sequence.active) sequence.stop("서버 연결이 끊겨 자동 진행을 중지했어요.");
      renderStage();
    },
    observe(payload){
      if (payload.stage1) acceptStage(payload.stage1);
      if (payload.stage2) acceptAlignment(payload.stage2);
      if (payload.stage3) acceptLift(payload.stage3);
      if (payload.stage4) acceptExit(payload.stage4);
      if (payload.stage5) acceptHeight(payload.stage5);
      if (payload.stage6) acceptInsertion(payload.stage6);
      if (payload.stage7) acceptRear(payload.stage7);
      if (payload.stage8) acceptFront(payload.stage8);
      renderWorkflow();
      if (selectedStage === 1 && stageState?.state === "MOVING"){
        const arrived = payload.joints.filter(state => state.arrived_now && state.command_id === stageState.command_ids[state.name]).length;
        $("workflow-message").textContent = `시작 자세로 이동 중 · 관절 도착 ${arrived}/7`;
      }
    },
  };
}
