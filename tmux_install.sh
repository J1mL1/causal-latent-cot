#!/usr/bin/env bash
set -e

DEPS_DIR="${TMUX_DEPS_DIR:-$HOME/tmux_deps}"

echo "[*] 进入依赖目录"
cd "$DEPS_DIR"

# echo "[*] 检查 deb 包是否齐全"
# required_pkgs=(
#   libevent-core
#   libtinfo6
#   libutempter0
#   tmux
# )

# for pkg in "${required_pkgs[@]}"; do
#   if ! ls ${pkg}_*.deb >/dev/null 2>&1; then
#     echo "[!] 缺少 ${pkg} 的 deb 包"
#     exit 1
#   fi
# done

echo "[*] 开始安装依赖"
dpkg -i libevent-core-*.deb
dpkg -i libtinfo6_*.deb
dpkg -i libutempter0_*.deb

echo "[*] 安装 tmux"
dpkg -i tmux_*.deb

echo "[*] 验证安装结果"
if command -v tmux >/dev/null 2>&1; then
  tmux -V
  echo "[✓] tmux 安装成功"
else
  echo "[✗] tmux 未正确安装"
  exit 1
fi
