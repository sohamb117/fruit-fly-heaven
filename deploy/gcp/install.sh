#!/bin/bash
# Run inside the extracted, verified deployment on a dedicated Debian 12 VM.
set -euo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1

if [[ $EUID -ne 0 ]]; then echo 'Run this installer with sudo.' >&2; exit 1; fi
: "${PUBLIC_ORIGIN:?Set PUBLIC_ORIGIN to https://PUBLIC_IP or https://DOMAIN}"
: "${CADDY_SHA256:?Set the verified SHA256 of the Caddy linux_amd64 release archive}"
CADDY_VERSION="${CADDY_VERSION:-2.11.4}"
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PACKAGE_ROOT PUBLIC_ORIGIN

python3 - <<'PY'
import os,re
from urllib.parse import urlsplit
value=urlsplit(os.environ['PUBLIC_ORIGIN'])
if value.scheme!='https' or not value.hostname or value.path or value.query or value.fragment or value.username or value.password or value.port is not None or not re.fullmatch(r'[A-Za-z0-9.-]+',value.hostname):
    raise SystemExit('Use an HTTPS domain or IPv4 origin, without port, path or credentials')
PY
if [[ ! "$CADDY_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ || ! "$CADDY_SHA256" =~ ^[a-f0-9]{64}$ ]]; then echo 'Invalid Caddy version or SHA256' >&2; exit 1; fi
if [[ "$(uname -m)" != x86_64 ]]; then echo 'This pinned Caddy archive targets an x86_64 E2 VM.' >&2; exit 1; fi
python3 "$PACKAGE_ROOT/tools/verify.py" "$PACKAGE_ROOT"
DEPLOYMENT_ID="$(python3 -c 'import json,os; print(json.load(open(os.environ["PACKAGE_ROOT"]+"/deployment-manifest.json"))["deploymentId"])')"
if [[ -f /var/lib/fly-training/coordinator.sqlite3 ]]; then
  python3 "$PACKAGE_ROOT/tools/verify.py" "$PACKAGE_ROOT" --database /var/lib/fly-training/coordinator.sqlite3
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends ca-certificates curl python3
getent group flytraining >/dev/null || groupadd --system flytraining
id flytraining >/dev/null 2>&1 || useradd --system --gid flytraining --home-dir /var/lib/fly-training --shell /usr/sbin/nologin flytraining
getent group caddy >/dev/null || groupadd --system caddy
id caddy >/dev/null 2>&1 || useradd --system --gid caddy --home-dir /var/lib/caddy --shell /usr/sbin/nologin caddy
install -d -m 0755 /opt/fly-training/releases /srv/fly-training/releases /etc/caddy /etc/fly-training
install -d -o flytraining -g flytraining -m 0750 /var/lib/fly-training
install -d -o caddy -g caddy -m 0750 /var/lib/caddy

TEMPORARY_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TEMPORARY_DIR"' EXIT
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  "https://github.com/caddyserver/caddy/releases/download/v${CADDY_VERSION}/caddy_${CADDY_VERSION}_linux_amd64.tar.gz" \
  --output "$TEMPORARY_DIR/caddy.tar.gz"
printf '%s  %s\n' "$CADDY_SHA256" "$TEMPORARY_DIR/caddy.tar.gz" | sha256sum --check --status
tar -xzf "$TEMPORARY_DIR/caddy.tar.gz" -C "$TEMPORARY_DIR" caddy LICENSE
install -m 0755 "$TEMPORARY_DIR/caddy" /usr/local/bin/caddy
install -d -m 0755 /usr/local/share/doc/caddy
install -m 0644 "$TEMPORARY_DIR/LICENSE" /usr/local/share/doc/caddy/LICENSE

RELEASE_ROOT="/opt/fly-training/releases/$DEPLOYMENT_ID"
WEB_ROOT="/srv/fly-training/releases/$DEPLOYMENT_ID"
install -d -m 0755 "$RELEASE_ROOT" "$WEB_ROOT"
cp -a "$PACKAGE_ROOT/server" "$PACKAGE_ROOT/config" "$PACKAGE_ROOT/tools" "$RELEASE_ROOT/"
cp "$PACKAGE_ROOT/deployment-manifest.json" "$RELEASE_ROOT/"
cp -a "$PACKAGE_ROOT/public/." "$WEB_ROOT/"
chown -R root:root "$RELEASE_ROOT" "$WEB_ROOT"
find "$RELEASE_ROOT" "$WEB_ROOT" -type d -exec chmod 0755 {} +
find "$RELEASE_ROOT" "$WEB_ROOT" -type f -exec chmod 0644 {} +

systemctl stop fly-training.service 2>/dev/null || true
if [[ -f "$PACKAGE_ROOT/state/bootstrap.sqlite3" && ! -e /var/lib/fly-training/coordinator.sqlite3 ]]; then
  install -o flytraining -g flytraining -m 0600 "$PACKAGE_ROOT/state/bootstrap.sqlite3" /var/lib/fly-training/coordinator.sqlite3
fi
ln -sfnT "$RELEASE_ROOT" /opt/fly-training/current
ln -sfnT "$WEB_ROOT" /srv/fly-training/current
printf 'PUBLIC_ORIGIN=%s\n' "$PUBLIC_ORIGIN" > /etc/fly-training/service.env
chmod 0644 /etc/fly-training/service.env
python3 - <<'PY'
import os
from pathlib import Path
from urllib.parse import urlsplit
origin=os.environ['PUBLIC_ORIGIN']
template=(Path(os.environ['PACKAGE_ROOT'])/'system/Caddyfile.template').read_text()
Path('/etc/caddy/Caddyfile').write_text(template.replace('@PUBLIC_ORIGIN@',origin).replace('@PUBLIC_HOST@',urlsplit(origin).hostname))
PY
/usr/local/bin/caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
install -m 0644 "$PACKAGE_ROOT/system/fly-training.service" /etc/systemd/system/fly-training.service
install -m 0644 "$PACKAGE_ROOT/system/caddy.service" /etc/systemd/system/caddy.service
systemctl daemon-reload
systemctl enable fly-training.service caddy.service
systemctl restart fly-training.service
for ATTEMPT in $(seq 1 20); do
  if curl --fail --silent http://127.0.0.1:7850/api/training/status > "$TEMPORARY_DIR/status.json"; then break; fi
  sleep 1
done
python3 - "$TEMPORARY_DIR/status.json" "$PACKAGE_ROOT/deployment-manifest.json" <<'PY'
import json,sys
status=json.load(open(sys.argv[1]));manifest=json.load(open(sys.argv[2]))
assert all(status[key]==manifest[key] for key in ('configHash','modelFingerprint')),'Coordinator identity mismatch'
print(json.dumps({key:status[key] for key in ('configHash','modelFingerprint','generation','acceptedResults')}))
PY
systemctl restart caddy.service
printf 'Installed %s at %s/train.html\n' "$DEPLOYMENT_ID" "$PUBLIC_ORIGIN"
printf 'Check TLS provisioning with: journalctl -u caddy -n 50 --no-pager\n'
