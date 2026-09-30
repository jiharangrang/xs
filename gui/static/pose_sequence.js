// 기존 단계 버튼과 토크 해제·R 잠금 버튼을 지정한 순서로 연결해요.
export function createStageSequence({startStage, readStage, stopStage, releaseBody, lockRight, stopAction, onChange = () => {}, setTimer = setTimeout, clearTimer = clearTimeout}){
  const completed = {1:"ARRIVED", 2:"ALIGNED", 3:"REACHED", 4:"REACHED", 5:"REACHED", 6:"REACHED", 7:"REACHED", 8:"REACHED"};
  const steps = [
    {stage:1, label:"1단계"},
    {delayMs:5000, label:"1단계 완료 뒤 5초 대기"},
    ...[2, 3, 4, 5, 6].flatMap(stage => [
      {stage, label:`${stage}단계`},
      {delayMs:2000, label:`${stage}단계 완료 뒤 2초 안정화`},
    ]),
    {action:"releaseBody", label:"몸통 토크 해제"},
    {action:"lockRight", label:"R 그리퍼 잠금"},
    {waitFor:"lockRight", delayMs:8000, label:"R 잠금 뒤 8초 대기"},
    {stage:7, label:"7단계 뒷그리퍼 당김"},
    {waitFor:"closeLeft", delayMs:8000, label:"L 닫기 뒤 8초 대기"},
    {action:"releaseBody", label:"몸통 토크 해제"},
    {stage:8, label:"8단계 앞그리퍼 전진"},
    {delayMs:2000, label:"8단계 완료 뒤 2초 안정화"},
    {action:"releaseBody", label:"몸통 토크 해제"},
  ];
  const actions = {releaseBody, lockRight};
  const idleMessage = "1·5초 → 2~6·각 2초 → 토크 해제 → R 잠금·8초 → 7·8초 → 토크 해제 → 8·2초 → 토크 해제";
  let run = null;
  let status = {state:"IDLE", stage:0, step:0, total:steps.length, action:null, message:idleMessage};

  function update(state, message, stage = run?.stage ?? status.stage){
    status = {state, stage, step:run?.step ?? status.step, total:steps.length, action:run?.action ?? null, message};
    onChange();
  }

  function stopCurrent(current){
    return current.action ? stopAction(current.action) : stopStage(current.stage);
  }

  function waitDelay(current, milliseconds){
    return new Promise(resolve => {
      current.finishWait = resolve;
      current.waitTimer = setTimer(() => {
        current.waitTimer = null;
        current.finishWait = null;
        resolve();
      }, milliseconds);
    });
  }

  async function execute(current){
    try {
      for (const [index, step] of steps.entries()){
        if (current.cancelled) break;
        current.step = index + 1;
        current.action = step.action ?? step.waitFor ?? null;
        if (step.stage) current.stage = step.stage;
        update("RUNNING", `${step.label} 실행 중…`);
        if (current.cancelled) break;
        if (step.delayMs){
          current.request = waitDelay(current, step.delayMs);
          await current.request;
          current.request = null;
          continue;
        }
        current.request = Promise.resolve().then(() => current.cancelled ? null :
          step.action ? actions[step.action]() : startStage(step.stage));
        let observed = await current.request;
        current.request = null;
        if (current.cancelled) break;
        if (step.action) continue;
        // 기존 버튼의 실행이 끝나고 그 버튼의 완료 상태가 나오면 다음으로 넘어가요.
        while (!current.cancelled && (observed?.active === true || step.stage === 1 && observed?.state === "MOVING")){
          await waitDelay(current, 200);
          if (current.cancelled) break;
          current.request = readStage(step.stage);
          observed = await current.request;
          current.request = null;
        }
        if (!current.cancelled && observed?.state !== completed[step.stage]){
          throw new Error(observed?.message || `${step.stage}단계가 완료되지 않았어요.`);
        }
      }
      if (!current.cancelled) update("COMPLETE", "자동 진행 완료 · 마지막 몸통 토크를 해제했어요.", 8);
    } catch (error){
      if (!current.cancelled){
        update("FAILED", `자동 진행을 중단했어요. ${error.message}`);
      }
    } finally {
      if (run === current && !current.stopTask){
        run = null;
        onChange();
      }
    }
  }

  function start(){
    if (run) return run.task;
    const current = {stage:1, step:0, action:null, cancelled:false, request:null, stopTask:null, waitTimer:null, finishWait:null};
    run = current;
    current.task = execute(current);
    return current.task;
  }

  function stop(message = "자동 진행을 중지했어요."){
    const current = run;
    if (!current) return Promise.resolve();
    if (current.stopTask) return current.stopTask;
    // 다음 단계 전송을 먼저 막고, 이미 보낸 시작 요청 뒤에 정지 요청을 보내요.
    current.cancelled = true;
    if (current.waitTimer !== null){
      clearTimer(current.waitTimer);
      current.waitTimer = null;
      current.finishWait?.();
      current.finishWait = null;
    }
    current.stopTask = Promise.resolve().then(async () => {
      await current.request?.catch(() => {});
      let stopError = null;
      try {
        const stopped = await stopCurrent(current);
        if (stopped?.stop_error) throw new Error(stopped.stop_error);
      }
      catch (error){ stopError = error; }
      await current.task;
      if (run === current) run = null;
      update(stopError ? "FAILED" : "STOPPED", stopError ? `${message} 정지 확인 실패: ${stopError.message}` : message, current.stage);
    });
    update("STOPPING", "자동 진행 중지 중…");
    return current.stopTask;
  }

  return {
    get active(){ return run !== null; },
    get status(){ return {...status}; },
    start,
    stop,
    reset(){
      if (!run){ status.step = 0; update("IDLE", idleMessage, 0); }
    },
  };
}
