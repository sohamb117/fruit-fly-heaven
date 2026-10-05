# GCP package verification — 2026-09-13

Artifact: [`fly-training-gcp-7e0bee5d6e65c55e.tar.gz`](../../dist/gcp/fly-training-gcp-7e0bee5d6e65c55e.tar.gz), 116,833,863 bytes, SHA256 `88db043a7c7915ab7709a4664325d33940737630e3f085629bea0e4ebf206715`.

- Deployment ID: `7e0bee5d6e65c55e`.
- Config SHA256: `1bad7d5c80d5129dcbe4f94b66fc725ec343053fb50bf70769c4a8e75510d00e`.
- Model fingerprint: `8bfd755893d540ed11e249c36ad7ae73d6a955c78bc57a3899ad18cc8c2bee13`.
- Source: refreshed `dist/training-client/fruit-fly-training-client-1bad7d5c80d5/` and consistent SQLite snapshot `coordinator-before-gcp.sqlite3` in this report directory.

The completed archive was independently decompressed into a temporary directory and all **140 payload file hashes** verified. The included database passed SQLite consistency and model/config checks. It preserves generation **1**, **8 completed jobs**, **8 pending jobs**, and no active leases. SQLite is packaged under private `state/`, outside the public web root. Public `index.html` redirects to `/train.html`.

Three packaging tests passed with `ResourceWarning` treated as an error: public allowlist/gzip equivalence/integrity, source mutation and symlink rejection, and consistent online SQLite backup preserving committed WAL transactions and model identity. `bash -n deploy/gcp/install.sh` passed. The complete IP HTTPS proxy configuration passed `caddy adapt --validate` under **Caddy v2.11.4**. These checks validate deployment packaging; they do not establish public TLS reachability or trained fly behavior.

The package contains read-only browser assets, the stdlib coordinator, canonical config, licenses and deployment helpers. It excludes the local Python web server, repository metadata, environment files and arbitrary repository directories. Gzip copies reduce actual browser transfer without requiring the small VM to compress the large model at request time; retaining original copies increases archive size.

Pinned Caddy archive: `caddy_2.11.4_linux_amd64.tar.gz`, 17,238,873 bytes, SHA256 `527fbf917c39189a1e3b31d34fa955601680b2d5c8055d2a87b8b9588dec7bb9`. Downloaded bytes matched the official SHA512 checksum and GitHub release asset SHA256 metadata. [Official release](https://github.com/caddyserver/caddy/releases/tag/v2.11.4).

The reserved deployment origin is `https://35.209.223.157`. Root deployment work performs provisioning, installation and public/browser verification separately. The source [`deploy/gcp/README.md`](../../deploy/gcp/README.md) was updated after this archive was built to replace its Caddy checksum placeholder with the verified value. The archive has not been rebuilt for that documentation-only update; its executable/config/model files remain the validated package described here.
