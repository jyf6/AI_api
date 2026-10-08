#!/usr/bin/env bash
set -euo pipefail

release_id="${1:?缺少发布版本号}"
archive="${2:?缺少发布包路径}"
[[ "$release_id" =~ ^[a-f0-9]{40}-[0-9]+-[0-9]+$ ]] || { echo '发布版本号格式错误' >&2; exit 1; }

root=/opt/app/chatgpt2api
release="$root/releases/$release_id"
current="$root/current"
previous=""
if [ -L "$current" ]; then
  previous="$(readlink -f "$current")"
fi

test -f /etc/flexi/chatgpt2api.env
test -f "$archive"
test ! -e "$release"
mkdir -p "$release"
tar -xzf "$archive" -C "$release"

cd "$release"
/home/ubuntu/.local/bin/uv sync --locked --no-dev --no-install-project --python /usr/bin/python3.12

# 依赖安装成功后再切换版本；旧版本保留以便失败时回退。
ln -sfn "$release" "$root/current.next"
mv -Tf "$root/current.next" "$current"

if sudo -n systemctl restart chatgpt2api; then
  for _ in $(seq 1 60); do
    if curl -fs -o /dev/null http://127.0.0.1:8010/health \
      && curl -fs -o /dev/null http://127.0.0.1:8010/api/proxies; then
      sudo -n systemctl enable chatgpt2api
      echo "Python 服务已发布：$release_id"
      exit 0
    fi
    sleep 2
  done
fi

echo '========= REMOTE SERVER LOGS =========' >&2
sudo -n journalctl -u chatgpt2api -n 100 --no-pager >&2 || true
echo '======================================' >&2
echo 'Python 服务检查失败，恢复上一版本' >&2
if [ -n "$previous" ]; then
  ln -sfn "$previous" "$root/current.rollback"
  mv -Tf "$root/current.rollback" "$current"
  sudo -n systemctl restart chatgpt2api || true
else
  sudo -n systemctl stop chatgpt2api || true
fi
exit 1
