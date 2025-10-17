#!/usr/bin/env bash
set -euo pipefail

# build_and_run.sh
# 编译 iOS App 并在 iPhone 15 模拟器上安装与启动
# 使用方式（示例）：
#   ./build_and_run.sh \
#     --scheme RealTimeDetectApp \
#     --project /path/to/YourApp.xcodeproj
# 或者使用 workspace：
#   ./build_and_run.sh \
#     --scheme RealTimeDetectApp \
#     --workspace /path/to/YourApp.xcworkspace
# 可选：指定 bundle id（否则从构建设置中解析）
#   ./build_and_run.sh --scheme RealTimeDetectApp --project ... --bundle-id com.example.RealTimeDetectApp

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

declare PROJECT=""
declare WORKSPACE=""
declare SCHEME=""
declare BUNDLE_ID=""
declare CONFIGURATION="Debug"
declare DEVICE_NAME="iPhone 15"

declare -r DEVICE_TYPE_PRIMARY="com.apple.CoreSimulator.SimDeviceType.iPhone-15"
declare -r DEVICE_TYPE_FALLBACK="com.apple.CoreSimulator.SimDeviceType.iPhone-15-Pro"

log() { echo "[INFO] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: $0 --scheme <SchemeName> [--project <.xcodeproj>|--workspace <.xcworkspace>] [--bundle-id <identifier>] [--configuration Debug|Release] [--device-name "iPhone 15"]
EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2;;
    --workspace) WORKSPACE="$2"; shift 2;;
    --scheme) SCHEME="$2"; shift 2;;
    --bundle-id) BUNDLE_ID="$2"; shift 2;;
    --configuration) CONFIGURATION="$2"; shift 2;;
    --device-name) DEVICE_NAME="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) die "Unknown option: $1";;
  esac
done

# 自动探测 xcodeproj/xcworkspace（若未指定）
auto_detect_project() {
  if [[ -z "$PROJECT" && -z "$WORKSPACE" ]]; then
    local ws
    ws=$(find "$ROOT_DIR" -maxdepth 3 -name "*.xcworkspace" -print -quit || true)
    local pj
    pj=$(find "$ROOT_DIR" -maxdepth 3 -name "*.xcodeproj" -print -quit || true)
    if [[ -n "$ws" ]]; then
      WORKSPACE="$ws"; log "Detected workspace: $WORKSPACE"
    elif [[ -n "$pj" ]]; then
      PROJECT="$pj"; log "Detected project: $PROJECT"
    else
      die "未找到 .xcworkspace 或 .xcodeproj。请使用 --project 或 --workspace 指定。"
    fi
  fi
}

# 选择最新可用 iOS Runtime
pick_ios_runtime() {
  local rid
  rid=$(xcrun simctl list runtimes | awk '/iOS / && /Identifier:/ && !/unavailable/ {rid=$NF} END {print rid}')
  [[ -z "$rid" ]] && die "未找到可用的 iOS Runtime。请打开 Xcode 安装模拟器运行时。"
  echo "$rid"
}

# 查找已存在的设备 UDID
find_device_udid() {
  local name="$1"
  local udid
  udid=$(xcrun simctl list devices | awk -v name="$name" '$0 ~ name && $0 ~ /\(.*\)/ {udid=$NF} END {gsub(/[()]/,"",udid); print udid}')
  echo "$udid"
}

# 如果设备不存在则创建
create_device_if_needed() {
  local name="$1"; local runtime_id="$2"
  local udid
  udid=$(find_device_udid "$name")
  if [[ -z "$udid" ]]; then
    log "未找到设备 \"$name\"，尝试创建（$DEVICE_TYPE_PRIMARY）..."
    if udid=$(xcrun simctl create "$name" "$DEVICE_TYPE_PRIMARY" "$runtime_id" 2>/dev/null); then
      log "已创建设备：$udid"
    else
      log "主类型创建失败，尝试回退（$DEVICE_TYPE_FALLBACK）..."
      udid=$(xcrun simctl create "$name" "$DEVICE_TYPE_FALLBACK" "$runtime_id")
      log "已创建设备（回退类型）：$udid"
    fi
  fi
  echo "$udid"
}

boot_device() {
  local udid="$1"
  log "启动设备：$udid"
  xcrun simctl boot "$udid" || true
  open -a Simulator || true
  # 等待设备完全启动
  xcrun simctl bootstatus "$udid" -b -s || true
}

# 编译 App
build_app() {
  local destination="platform=iOS Simulator,id=$1"
  local args=( -scheme "$SCHEME" -configuration "$CONFIGURATION" -sdk iphonesimulator -destination "$destination" build )
  if [[ -n "$WORKSPACE" ]]; then
    args=( -workspace "$WORKSPACE" "${args[@]}" )
  else
    args=( -project "$PROJECT" "${args[@]}" )
  fi
  log "开始编译：xcodebuild ${args[*]}"
  xcodebuild "${args[@]}"
}

# 获取构建设置
get_build_settings() {
  local settings_file="$(mktemp)"
  local args=( -scheme "$SCHEME" -configuration "$CONFIGURATION" -sdk iphonesimulator -showBuildSettings )
  if [[ -n "$WORKSPACE" ]]; then
    args=( -workspace "$WORKSPACE" "${args[@]}" )
  else
    args=( -project "$PROJECT" "${args[@]}" )
  fi
  xcodebuild "${args[@]}" > "$settings_file"
  local target_build_dir wrapper_name product_bundle_id
  target_build_dir=$(awk -F' = ' '/TARGET_BUILD_DIR/ {print $2; exit}' "$settings_file")
  wrapper_name=$(awk -F' = ' '/WRAPPER_NAME/ {print $2; exit}' "$settings_file")
  product_bundle_id=$(awk -F' = ' '/PRODUCT_BUNDLE_IDENTIFIER/ {print $2; exit}' "$settings_file")
  [[ -z "$target_build_dir" || -z "$wrapper_name" ]] && die "解析构建路径失败。"
  echo "$target_build_dir/$wrapper_name|${BUNDLE_ID:-$product_bundle_id}"
}

# 安装并启动 App
install_and_launch() {
  local udid="$1"; local app_path="$2"; local bundle_id="$3"
  [[ -z "$bundle_id" ]] && die "无法确定 Bundle Identifier，请使用 --bundle-id 指定。"
  log "安装 App：$app_path"
  xcrun simctl install "$udid" "$app_path"
  log "启动 App：$bundle_id"
  xcrun simctl launch -w "$udid" "$bundle_id"
}

# 主流程
[[ -z "$SCHEME" ]] && { usage; die "必须指定 --scheme"; }
auto_detect_project
RUNTIME_ID=$(pick_ios_runtime)
UDID=$(create_device_if_needed "$DEVICE_NAME" "$RUNTIME_ID")
boot_device "$UDID"
build_app "$UDID"
SETTINGS=$(get_build_settings)
APP_PATH="${SETTINGS%%|*}"
BUNDLE="${SETTINGS##*|}"
install_and_launch "$UDID" "$APP_PATH" "$BUNDLE"

log "完成：已在 $DEVICE_NAME 上启动应用（$BUNDLE）。"