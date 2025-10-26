#!/usr/bin/env bash
set -euo pipefail

APP_NAME="RealTimeDetectApp"
SCHEME="RealTimeDetectApp"
PROJECT="RealTimeDetectApp.xcodeproj"
BUNDLE_ID="com.yoloset.RealTimeDetectApp"
DEVICE_NAME="iPhone 16"
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
  local target="com.apple.CoreSimulator.SimRuntime.iOS-26-0"
  local available=""
  if command -v python3 >/dev/null 2>&1; then
    available=$( { xcrun simctl list runtimes --json | python3 - <<'PY'
import sys, json
try:
    data = json.load(sys.stdin)
except Exception:
    print(""); sys.exit(0)
for rt in data.get("runtimes", []):
    if rt.get("identifier") == "com.apple.CoreSimulator.SimRuntime.iOS-26-0" and rt.get("available", True):
        print("1"); sys.exit(0)
print("")
PY
    } 2>/dev/null || echo "" )
  fi
  if [[ "${available}" == "1" ]]; then
    echo "${target}"; return 0
  fi
  if xcrun simctl list runtimes 2>/dev/null | grep -q "com.apple.CoreSimulator.SimRuntime.iOS-26-0"; then
    echo "${target}"; return 0
  fi
  err "未找到 iOS 26.0 Runtime（com.apple.CoreSimulator.SimRuntime.iOS-26-0）。请在 Xcode -> Settings -> Platforms 中安装。"
  echo ""
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
    if 'iOS-26-0' not in rt:
        continue
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
    # 文本模式无法安全过滤运行时，返回空以便后续创建 26.0 设备
    udid=""
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
  udid=$(xcrun simctl create "${name}" com.apple.CoreSimulator.SimDeviceType.iPhone-16 "${runtime}" 2>/dev/null || true)
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