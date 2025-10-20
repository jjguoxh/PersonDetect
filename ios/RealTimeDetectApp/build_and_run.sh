#!/usr/bin/env bash
set -euo pipefail

APP_NAME="RealTimeDetectApp"
SCHEME="RealTimeDetectApp"
PROJECT="RealTimeDetectApp.xcodeproj"
BUNDLE_ID="com.yoloset.RealTimeDetectApp"
DEVICE_NAME="iPhone 15"
DERIVED_DATA="build"

log() { echo "[INFO] $*" >&2; }
warn() { echo "[WARN] $*" >&2; }
err() { echo "[ERROR] $*" >&2; }

ensure_developer_dir() {
  local current
  current=$(xcode-select -p 2>/dev/null || true)
  if [[ -z "${current}" ]]; then
    export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
    log "自动设置 DEVELOPER_DIR=${DEVELOPER_DIR}"
  else
    log "使用 Xcode Developer 目录：${current}"
  fi
}

pick_ios_runtime() {
  local runtime=""
  if [[ -n "${IOS_RUNTIME_IDENTIFIER:-}" ]]; then
    echo "${IOS_RUNTIME_IDENTIFIER}"; return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    runtime=$( { xcrun simctl list runtimes --json | python3 - <<'PY'
import sys, json, re
try:
    data = json.load(sys.stdin)
except Exception:
    print(""); sys.exit(0)
rts = [r for r in data.get('runtimes', []) if r.get('platform') == 'iOS' and r.get('available', True) and str(r.get('identifier','')).startswith('com.apple.CoreSimulator.SimRuntime.iOS')]

def verkey(rt):
    m = re.search(r'iOS-(\d+)-(\d+)', rt.get('identifier',''))
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r'iOS-(\d+)', rt.get('identifier',''))
    if m:
        return (int(m.group(1)), 0)
    return (0, 0)
rts.sort(key=verkey, reverse=True)
print(rts[0]['identifier'] if rts else "")
PY
    } 2>/dev/null || echo "" )
  fi
  if [[ -z "${runtime}" ]]; then
    runtime=$(xcrun simctl list runtimes 2>/dev/null | awk '/^iOS / && !/unavailable/ && $NF ~ /com.apple.CoreSimulator.SimRuntime.iOS-/ {id=$NF} END{print id}')
  fi
  echo "${runtime}"
}

find_device_udid() {
  local name="$1"
  local udid=""
  if command -v python3 >/dev/null 2>&1; then
    udid=$( { xcrun simctl list devices --json | python3 - "$name" <<'PY'
import sys, json
name = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception:
    print(""); sys.exit(0)
booted = None
first = None
for rt, devs in data.get('devices', {}).items():
    for d in devs:
        if d.get('name') == name and 'udid' in d:
            if first is None:
                first = d['udid']
            if d.get('state') == 'Booted':
                booted = d['udid']
                break
    if booted:
        break
print(booted or first or "")
PY
    } 2>/dev/null || echo "" )
  fi
  if [[ -z "${udid}" ]]; then
    udid=$(xcrun simctl list devices 2>/dev/null | sed -nE "/${name}/s/.*\([^)]+\)\s+\(([A-F0-9-]+)\).*/\1/p" | head -n 1)
  fi
  echo "${udid}"
}

boot_and_show_simulator() {
  local udid="$1"
  local state
  state=$( { xcrun simctl list devices --json | python3 - "$udid" <<'PY'
import sys, json
udid = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception:
    print(""); sys.exit(0)
for _, devs in data.get('devices', {}).items():
    for d in devs:
        if d.get('udid') == udid:
            print(d.get('state', ''))
            sys.exit(0)
print("")
PY
  } 2>/dev/null || echo "" )
  if [[ "${state}" != "Booted" ]]; then
    log "启动设备：${udid}"
    xcrun simctl boot "${udid}" || true
    xcrun simctl bootstatus "${udid}" -b || true
  fi
  log "打开 Simulator 应用"
  open -a Simulator || true
}

build_install_launch() {
  local udid="$1"
  log "开始构建：scheme=${SCHEME}，目标设备=${udid}"
  if command -v xcpretty >/dev/null 2>&1; then
    xcodebuild \
      -project "${PROJECT}" \
      -scheme "${SCHEME}" \
      -configuration Debug \
      -derivedDataPath "${DERIVED_DATA}" \
      -destination "platform=iOS Simulator,id=${udid}" \
      CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO \
      build | xcpretty || true
  else
    xcodebuild \
      -project "${PROJECT}" \
      -scheme "${SCHEME}" \
      -configuration Debug \
      -derivedDataPath "${DERIVED_DATA}" \
      -destination "platform=iOS Simulator,id=${udid}" \
      CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO \
      build || true
  fi

  local app_path="${DERIVED_DATA}/Build/Products/Debug-iphonesimulator/${APP_NAME}.app"
  if [[ ! -d "${app_path}" ]]; then
    err "未找到构建产物：${app_path}"; exit 1
  fi
  log "安装 App：${app_path}"
  xcrun simctl install "${udid}" "${app_path}" || true

  local bundle_id
  bundle_id=$(/usr/libexec/PlistBuddy -c 'Print CFBundleIdentifier' "${app_path}/Info.plist" 2>/dev/null || true)
  if [[ -z "${bundle_id}" ]]; then bundle_id="${BUNDLE_ID}"; fi
  log "启动 App：${bundle_id}"
  xcrun simctl launch "${udid}" "${bundle_id}" || true
}

create_device_if_needed() {
  local name="$1"
  local udid
  udid=$(find_device_udid "$name")
  if [[ -n "${udid}" ]]; then
    log "找到现有设备：${name} (${udid})"
    echo "${udid}"; return 0
  fi
  local runtime
  runtime=$(pick_ios_runtime)
  if [[ -z "${runtime}" ]]; then
    err "未找到可用的 iOS Runtime"; exit 1
  fi
  log "创建设备：${name}，runtime=${runtime}"
  udid=$(xcrun simctl create "${name}" com.apple.CoreSimulator.SimDeviceType.iPhone-15 "${runtime}" 2>/dev/null || true)
  if [[ -z "${udid}" ]]; then
    err "创建设备失败：${name}，runtime=${runtime}"; exit 1
  fi
  log "创建设备成功：${udid}"
  echo "${udid}"
}

main() {
  ensure_developer_dir
  local udid
  udid=$(create_device_if_needed "${DEVICE_NAME}")
  boot_and_show_simulator "${udid}"
  build_install_launch "${udid}"
  log "完成：已在 ${DEVICE_NAME} 启动 ${APP_NAME}"
}

main "$@"