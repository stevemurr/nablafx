#!/usr/bin/env bash
#
# Build a per-model .clap bundle from an export staging directory.
#
# Usage:
#   build.sh <staging_dir> <out.clap>
#
# Inputs:
#   <staging_dir>   directory with model.onnx + plugin_meta.json
#                   (produced by `nablafx-export`)
#   <out.clap>      destination bundle path; ends in .clap by convention
#
# Behavior:
#   - On first run (or after cmake inputs change) configures and builds the
#     generic dylib under native/clap/build/.
#   - Copies the dylib, the onnxruntime dylib, model.onnx, and plugin_meta.json
#     into the output .clap bundle with the expected Mac layout.
#   - Ad-hoc codesigns the bundle (`codesign --force --deep --sign -`). This
#     is enough for local dev; distribution needs a real identity + notarize.
set -eu

usage() {
    cat <<EOF >&2
usage: $(basename "$0") <staging_dir> <out.clap>
EOF
    exit 2
}

if [ "$#" -ne 2 ]; then
    usage
fi

if [ "$(uname -s)" != "Darwin" ]; then
    echo "build.sh: this step must run on macOS (arm64)" >&2
    exit 1
fi

STAGING="$(cd "$1" && pwd)"
OUT="$2"

if [ ! -f "$STAGING/model.onnx" ] || [ ! -f "$STAGING/plugin_meta.json" ]; then
    echo "error: $STAGING is missing model.onnx or plugin_meta.json" >&2
    exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$HERE/build"

# Configure + build the dylib once.
if [ ! -f "$BUILD_DIR/build_config.sh" ] || [ ! -f "$BUILD_DIR/nablafx_clap.so" ]; then
    cmake -S "$HERE" -B "$BUILD_DIR" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" -j
fi

# shellcheck disable=SC1091
. "$BUILD_DIR/build_config.sh"

# Pull model_id / effect_name from plugin_meta.json using python3 (macOS ships
# python3 via CLT; fall back to `plutil` or `grep` if needed).
pyscript='
import json, sys
m = json.load(open(sys.argv[1]))
print(m["model_id"])
print(m["effect_name"])
print(m["model_id"])  # used as bundle executable too
'
PYOUT=$(/usr/bin/env python3 -c "$pyscript" "$STAGING/plugin_meta.json")
MODEL_ID=$(printf '%s\n' "$PYOUT" | sed -n '1p')
EFFECT_NAME=$(printf '%s\n' "$PYOUT" | sed -n '2p')
EXECUTABLE=$(printf '%s\n' "$PYOUT" | sed -n '3p')

# Layout:
#   <OUT>/Contents/Info.plist
#   <OUT>/Contents/MacOS/<executable>
#   <OUT>/Contents/Frameworks/libonnxruntime.<ver>.dylib
#   <OUT>/Contents/Frameworks/libonnxruntime.dylib (symlink)
#   <OUT>/Contents/Resources/model.onnx
#   <OUT>/Contents/Resources/plugin_meta.json
rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Frameworks" "$OUT/Contents/Resources"

cp "$NABLAFX_CLAP_DYLIB"     "$OUT/Contents/MacOS/$EXECUTABLE"
cp "$NABLAFX_CLAP_ORT_DYLIB" "$OUT/Contents/Frameworks/"
ln -sf "$(basename "$NABLAFX_CLAP_ORT_DYLIB")" \
       "$OUT/Contents/Frameworks/libonnxruntime.dylib"

cp "$STAGING/model.onnx"        "$OUT/Contents/Resources/"
cp "$STAGING/plugin_meta.json"  "$OUT/Contents/Resources/"

BUNDLE_ID="com.nablafx.$MODEL_ID"
sed \
    -e "s|__BUNDLE_EXECUTABLE__|$EXECUTABLE|g" \
    -e "s|__BUNDLE_IDENTIFIER__|$BUNDLE_ID|g" \
    -e "s|__BUNDLE_NAME__|$EFFECT_NAME|g" \
    "$HERE/template/Info.plist.in" > "$OUT/Contents/Info.plist"

# Ad-hoc codesign for local dev. Distribution builds need a real identity.
codesign --force --deep --sign - "$OUT"

echo "built $OUT"
echo "  effect_name: $EFFECT_NAME"
echo "  model_id:    $MODEL_ID"
echo "  bundle_id:   $BUNDLE_ID"
