// 실제 장치 없이 기존 버튼 연결의 순서와 고정 대기 및 사용자 중지를 검증해요.
import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import {createStageSequence} from "../gui/static/pose_sequence.js";
import {createWorkflowPanel} from "../gui/static/pose_workflow.js";

const flush = () => new Promise(resolve => setImmediate(resolve));
const moving = stage => ({stage, state:stage === 1 ? "MOVING" : "OBSERVING", ...(stage === 1 ? {} : {active:true})});
const finished = stage => ({stage, state:stage === 1 ? "ARRIVED" : stage === 2 ? "ALIGNED" : "REACHED", active:false});
const order = [...[1, 2, 3, 4, 5, 6].map(stage => `stage${stage}`), "release-body", "lock-right", "stage7", "release-body", "stage8", "release-body"];

function fixture(options = {}){
  const commands = [], stopped = [], waits = [];
  const sequence = createStageSequence({
    startStage(stage){ commands.push(`stage${stage}`); return moving(stage); },
    async readStage(stage){ return finished(stage); },
    async stopStage(stage){ stopped.push(stage); },
    async releaseBody(){ commands.push("release-body"); },
    async lockRight(){ commands.push("lock-right"); },
    async stopAction(action){ stopped.push(action); },
    setTimer(callback, milliseconds){ waits.push(milliseconds); queueMicrotask(callback); return 0; },
    clearTimer(){},
    ...options,
  });
  return {sequence, commands, stopped, waits};
}

test("기존 버튼 순서에 단계별 2초 안정화와 기존 5초 대기를 연결해요", async () => {
  const {sequence, commands, stopped, waits} = fixture();
  const task = sequence.start();
  assert.equal(sequence.start(), task);
  await task;
  assert.deepEqual(commands, order);
  assert.deepEqual(waits.filter(milliseconds => milliseconds !== 200), [5000, 2000, 2000, 2000, 2000, 2000, 8000, 8000, 2000]);
  assert.deepEqual(stopped, []);
  assert.equal(sequence.status.state, "COMPLETE");
  assert.equal(sequence.active, false);
});

test("기존 버튼의 완료 상태가 나올 때까지 다음 버튼을 실행하지 않아요", async () => {
  const reads = new Map();
  const {sequence, commands, stopped} = fixture({
    async readStage(stage){
      const count = (reads.get(stage) ?? 0) + 1;
      reads.set(stage, count);
      return count < 3 ? {...finished(stage), active:true, ...(stage === 1 ? {state:"MOVING"} : {})} : {
        ...finished(stage),
        stop_error:"기존 버튼 표시", camera_error:"기존 버튼 표시",
      };
    },
  });
  await sequence.start();
  assert.deepEqual(commands, order);
  assert.equal(reads.get(1), 3);
  assert.equal(reads.get(8), 3);
  assert.deepEqual(stopped, []);
  assert.equal(sequence.status.state, "COMPLETE");
});

test("버튼 응답이 이미 완료 상태이면 추가 도착 검사 없이 다음으로 넘어가요", async () => {
  const commands = [];
  const {sequence} = fixture({
    startStage(stage){ commands.push(stage); return finished(stage); },
    readStage(){ assert.fail("종료된 버튼을 다시 조회하면 안 돼요."); },
  });
  await sequence.start();
  assert.deepEqual(commands, [1, 2, 3, 4, 5, 6, 7, 8]);
});

for (const state of ["FAILED", "STOPPED", "IDLE"]){
  test(`${state}는 기존 버튼의 완료가 아니므로 다음 버튼을 실행하지 않아요`, async () => {
    const {sequence, commands, stopped} = fixture({readStage:async () => ({state, active:false, message:"기존 버튼이 완료되지 않았어요."})});
    await sequence.start();
    assert.deepEqual(commands, ["stage1"]);
    assert.deepEqual(stopped, []);
    assert.equal(sequence.status.state, "FAILED");
  });
}

test("재실행도 첫 버튼부터 같은 순서를 한 번씩 실행해요", async () => {
  const {sequence, commands} = fixture();
  await sequence.start();
  await sequence.start();
  assert.deepEqual(commands, [...order, ...order]);
});

test("1단계 완료 뒤 5초가 지나야 2단계를 실행해요", async () => {
  const timers = [];
  const {sequence, commands, stopped} = fixture({
    setTimer(callback, milliseconds){ const timer = {callback, milliseconds}; timers.push(timer); return timer; },
  });
  const task = sequence.start();
  await flush();
  assert.deepEqual(commands, ["stage1"]);
  assert.equal(timers[0].milliseconds, 200);
  timers[0].callback();
  await flush();
  assert.deepEqual(commands, ["stage1"]);
  assert.equal(timers[1].milliseconds, 5000);
  assert.match(sequence.status.message, /1단계 완료 뒤 5초/);
  timers[1].callback();
  await flush();
  assert.deepEqual(commands, ["stage1", "stage2"]);
  await sequence.stop();
  await task;
  assert.deepEqual(stopped, [2]);
});

test("1단계 완료 뒤 대기 중 정지하면 2단계를 실행하지 않아요", async () => {
  let timer;
  const {sequence, commands, stopped} = fixture({
    startStage(stage){ commands.push(`stage${stage}`); return finished(stage); },
    setTimer(callback, milliseconds){ timer = {callback, milliseconds, cleared:false}; return timer; },
    clearTimer(value){ value.cleared = true; },
  });
  const task = sequence.start();
  await flush();
  assert.equal(timer.milliseconds, 5000);
  await sequence.stop();
  await task;
  assert.equal(timer.cleared, true);
  assert.deepEqual(commands, ["stage1"]);
  assert.deepEqual(stopped, [1]);
});

test("그리퍼 잠금 대기 중 정지하면 대기를 취소하고 다음 버튼을 실행하지 않아요", async () => {
  const timers = [];
  const {sequence, commands, stopped} = fixture({
    startStage(stage){ commands.push(`stage${stage}`); return finished(stage); },
    setTimer(callback, milliseconds){
      if (milliseconds === 2000){ queueMicrotask(callback); return 0; }
      const timer = {callback, milliseconds, cleared:false}; timers.push(timer); return timer;
    },
    clearTimer(timer){ timer.cleared = true; },
  });
  const task = sequence.start();
  await flush();
  assert.equal(commands.at(-1), "stage1");
  assert.equal(timers[0].milliseconds, 5000);
  timers[0].callback();
  await flush();
  assert.equal(commands.at(-1), "lock-right");
  assert.equal(timers[1].milliseconds, 8000);
  timers[1].callback();
  await flush();
  assert.equal(commands.at(-1), "stage7");
  assert.equal(timers[2].milliseconds, 8000);
  await sequence.stop();
  await task;
  assert.equal(timers[2].cleared, true);
  assert.deepEqual(stopped, ["closeLeft"]);
  assert.equal(commands.includes("stage8"), false);
});

test("2초 안정화가 끝나기 전에는 다음 단계나 토크 해제를 실행하지 않아요", async () => {
  for (const stage of [2, 6, 8]){
    let resume;
    const {sequence, commands} = fixture({
      startStage(value){ commands.push(`stage${value}`); return finished(value); },
      setTimer(callback, milliseconds){
        if (milliseconds === 2000 && sequence.status.stage === stage) resume = callback;
        else queueMicrotask(callback);
        return 0;
      },
    });
    const task = sequence.start();
    await flush();
    assert.equal(commands.at(-1), `stage${stage}`);
    assert.match(sequence.status.message, /2초 안정화/);
    resume();
    await task;
    assert.deepEqual(commands, order);
  }
});

test("시작 요청 중 정지하면 그 응답 뒤에 정지하고 다음 버튼을 막아요", async () => {
  let finishStart;
  const events = [];
  const {sequence} = fixture({
    startStage(stage){ events.push(`start-${stage}`); return new Promise(resolve => { finishStart = resolve; }); },
    async stopStage(stage){ events.push(`stop-${stage}`); },
  });
  const task = sequence.start();
  await flush();
  const stopping = sequence.stop();
  assert.equal(sequence.stop(), stopping);
  await flush();
  assert.deepEqual(events, ["start-1"]);
  finishStart(moving(1));
  await stopping;
  await task;
  assert.deepEqual(events, ["start-1", "stop-1"]);
  assert.equal(sequence.status.state, "STOPPED");
});

test("상태 조회 중 정지해도 뒤의 버튼을 실행하지 않아요", async () => {
  let finishRead;
  const {sequence, commands, stopped} = fixture({readStage:() => new Promise(resolve => { finishRead = resolve; })});
  const task = sequence.start();
  await flush();
  const stopping = sequence.stop();
  finishRead(finished(1));
  await stopping;
  await task;
  assert.deepEqual(commands, ["stage1"]);
  assert.deepEqual(stopped, [1]);
});

function panelFixture(t, options = {}){
  const elements = new Map(), events = new Map(), commands = [], notices = [];
  const original = {document:globalThis.document, fetch:globalThis.fetch, addEventListener:globalThis.addEventListener};
  const element = id => {
    if (!elements.has(id)) elements.set(id, {disabled:false, dataset:{}, textContent:"", value:"100", addEventListener(event, handler){ this[event] = handler; }, removeAttribute(){}, reportValidity(){ return true; }});
    return elements.get(id);
  };
  globalThis.document = {getElementById:element, querySelectorAll:() => []};
  globalThis.addEventListener = (event, handler) => events.set(event, handler);
  globalThis.fetch = async url => ({ok:true, json:async () => {
    const stage = Number(url.match(/stage(\d)/)?.[1]);
    return {...finished(stage), ...(stage === 7 ? {side_return_completed:true} : {}), ...options.observed?.(stage)};
  }});
  t.after(() => Object.assign(globalThis, original));
  const panel = createWorkflowPanel({
    server:"http://mock.invalid", onChange(){}, onTargets(){}, notify(message){ notices.push(message); },
    async releaseBody(){ commands.push({route:"release-body"}); },
    async lockRight(){ commands.push({route:"lock-right"}); },
    async stopAction(action){ commands.push({route:`stop-${action}`}); },
    setTimer(callback){ queueMicrotask(callback); return 0; },
    clearTimer(){},
    async post(route, body){
      if (route === "/api/camera/start") return {state:"STOPPED", message:"장치 없는 테스트"};
      commands.push({route, body});
      const stage = Number(route.match(/stage(\d)/)[1]);
      if (options.failStart === stage && route.endsWith("start")) throw new Error("기존 버튼 요청 오류");
      return route.endsWith("stop") ? {state:"STOPPED", active:false} : {...moving(stage), run_id:`started-${stage}-${commands.length}`, ...options.startStatus?.(stage)};
    },
  });
  return {panel, element, commands, events, notices};
}

test("개별 버튼과 자동 진행이 같은 실행 함수와 거리값을 사용해요", async t => {
  const {panel, element, commands} = panelFixture(t);
  await flush();
  panel.setContext({connected:true, busy:false});
  element("stage7-distance").value = "40";
  element("stage8-distance").value = "50";
  for (let stage = 1; stage <= 8; stage++){
    await element(`stage${stage}-start`).click();
    panel.observe({[`stage${stage}`]:{...finished(stage), side_return_completed:true}, joints:[]});
  }
  const manual = commands.splice(0);
  element("sequence-start").click();
  await flush();
  const automatic = commands.filter(command => command.route.startsWith("/api/stage"));
  assert.deepEqual(automatic, manual);
  assert.deepEqual(commands.map(command => command.route), order.map(command => command.startsWith("stage") ? `/api/${command}/start` : command));
  assert.deepEqual(automatic[6].body, {distance_mm:40});
  assert.deepEqual(automatic[7].body, {distance_mm:50});
  assert.equal(panel.busy, false);
});

test("재시작 응답의 거리값과 비어 있는 각도가 섞여도 자동 진행을 끝까지 연결해요", async t => {
  const {panel, element, commands, notices} = panelFixture(t, {startStatus:stage => {
    if (stage === 5) return {remaining_mm:12, side_clearance_mm:null, tilt_deg:null};
    if (stage === 6) return {remaining_mm:22, height_error_mm:null, height_tolerance_mm:2, tilt_deg:null};
    return {};
  }});
  await flush();
  panel.setContext({connected:true, busy:false});
  element("sequence-start").click();
  await flush();
  assert.deepEqual(commands.map(command => command.route), order.map(command => command.startsWith("stage") ? `/api/${command}/start` : command));
  assert.deepEqual(notices, []);
  assert.equal(panel.busy, false);
});

test("상태 스트림의 누락된 측정값은 표시를 생략하고 실제 0은 표시해요", async t => {
  const {panel, element} = panelFixture(t);
  await flush();
  panel.setContext({connected:true, busy:false});
  for (const missing of [null, undefined, NaN]){
    assert.doesNotThrow(() => panel.observe({stage5:{...moving(5), run_id:"stream-5", message:"높이 관측 중", remaining_mm:12, side_clearance_mm:missing, tilt_deg:missing}, joints:[]}));
    assert.doesNotThrow(() => panel.observe({stage6:{...moving(6), run_id:"stream-6", message:"삽입 관측 중", remaining_mm:22, height_error_mm:missing, height_tolerance_mm:2, tilt_deg:missing}, joints:[]}));
    assert.doesNotMatch(element("workflow-message").textContent, /높이 오차|정면 오차/);
  }
  panel.observe({stage6:{...moving(6), run_id:"stream-6", message:"삽입 관측 중", remaining_mm:0, height_error_mm:0, tilt_deg:0}, joints:[]});
  assert.match(element("workflow-message").textContent, /높이 오차 0.00 mm/);
  assert.match(element("workflow-message").textContent, /정면 오차 0.00°/);
});

test("화면의 1단계 도착 실패를 완료로 취급하지 않아요", async t => {
  const {panel, element, commands} = panelFixture(t, {observed:stage => stage === 1 ? {state:"FAILED", message:"제한 시간 안에 모든 관절의 현재 도착을 확인하지 못했습니다."} : {}});
  await flush();
  panel.setContext({connected:true, busy:false});
  element("sequence-start").click();
  await flush();
  assert.deepEqual(commands.map(command => command.route), ["/api/stage1/start"]);
  assert.match(element("sequence-action").textContent, /제한 시간 안에/);
  assert.equal(panel.busy, false);
});

test("요청 실패도 완료가 아니므로 다음 버튼을 보내지 않아요", async t => {
  const {panel, element, commands, notices} = panelFixture(t, {failStart:3});
  await flush();
  panel.setContext({connected:true, busy:false});
  element("sequence-start").click();
  await flush();
  assert.equal(commands.some(command => command.route === "/api/stage4/start"), false);
  assert.deepEqual(notices, ["기존 버튼 요청 오류"]);
  assert.equal(commands.some(command => command.route.endsWith("/stop")), false);
});

for (const interruption of ["disconnect", "pagehide"]){
  test(`${interruption}이면 사용자 중지처럼 후속 버튼을 취소해요`, async t => {
    const {panel, element, commands, events} = panelFixture(t);
    await flush();
    panel.setContext({connected:true, busy:false});
    element("sequence-start").click();
    if (interruption === "disconnect") panel.setContext({connected:false, busy:false});
    else events.get("pagehide")();
    await flush();
    assert.equal(commands.some(command => command.route === "/api/stage2/start"), false);
    assert.match(element("sequence-action").textContent, /중지/);
  });
}

test("공유 토크 해제 함수는 기존 J1~J7 버튼 명령만 보내요", async () => {
  const html = readFileSync(new URL("../gui/static/pose.html", import.meta.url), "utf8");
  const source = html.slice(html.indexOf("async function releaseBodyTorque()"), html.indexOf('$("release-body").addEventListener'));
  const sent = [];
  const releaseBody = runInNewContext(source + "\nreleaseBodyTorque", {
    ARM:["J1", "J2", "J3", "J4", "J5", "J6", "J7"],
    async post(route, body){ sent.push({route, joint:body.joint, enabled:body.enabled}); return {ok:true}; },
  });
  await releaseBody();
  assert.deepEqual(sent, ["J1", "J2", "J3", "J4", "J5", "J6", "J7"].map(joint => ({route:"/api/torque", joint, enabled:false})));
});

test("자동 R 잠금도 기존 자세 버튼의 명령을 그대로 보내요", async () => {
  const html = readFileSync(new URL("../gui/static/pose.html", import.meta.url), "utf8");
  const source = html.slice(html.indexOf("async function sendPose("), html.indexOf("function observeMotion("));
  const sent = [];
  const sendPose = runInNewContext(source + "\nsendPose", {
    connected:true, pendingRequest:null, activeMove:null, stopping:false, scanBusy:false, scanRequest:false,
    recalling:false, workflow:{busy:true}, GRIP:["G_L", "G_R"], angles:{},
    $:() => ({textContent:""}), updateButtons(){}, finishMove(){}, toast(){},
    fmt:value => String(value), THREE:{MathUtils:{degToRad:value => value}}, applyAngles(){}, paint(){},
    setTimeout(){ return 0; },
    async post(route, body){ sent.push({route, angles:body.angles_deg, remember:body.remember}); return {command_ids:{G_R:42}}; },
  });
  await sendPose({G_R:4.6}, false, true);
  assert.deepEqual(sent, [{route:"/api/pose", angles:{G_R:4.6}, remember:true}]);
});
