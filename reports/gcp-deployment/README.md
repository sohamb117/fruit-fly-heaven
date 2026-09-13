# Live GCP training deployment

Launched 2026-09-13 at **https://35.209.223.157/train.html**. This deployment hosts the training UI and shared coordinator. Contributor browsers execute the BANC v888 neural model and native FlyBody simulation locally; no cloud GPU is provisioned.

## Resources

- Project: `monereko-20260809`
- VM: `fly-training-1`, zone `us-central1-a`, `e2-micro`, Debian 12, no service account
- Disk: `fly-training-1`, 10 GB `pd-standard`, retained if the VM is deleted
- Static external IPv4: `fly-training-ip`, `35.209.223.157`, explicitly **STANDARD** network tier
- VPC: `fly-training-net`; subnet `fly-training-uscentral1`, `10.89.0.0/24`
- Public ingress: TCP 80/443; SSH restricted to IAP range `35.235.240.0/20`
- Coordinator: systemd `fly-training`, loopback port 7850, database `/var/lib/fly-training/coordinator.sqlite3`
- HTTPS: Caddy 2.11.4, Let's Encrypt public IP certificate, automatic renewal; persistent certificate state under `/var/lib/caddy`
- Exact release: `7e0bee5d6e65c55e`; artifact/checksums in [packaging.md](packaging.md)

Full inventory is recorded in `instance.json`, `disk.json`, `firewall.json` and `server-health.txt`. The coordinator and Caddy admin ports bind only to loopback. Public files are copied from a verified allowlist; the repository, database and credentials are outside the web root.

## Verified outcome

- Trusted HTTPS without certificate overrides; HTTP redirects to HTTPS.
- Sixteen HTTP checks passed: root entry, UI, matching model identity, both WASM MIME types and precompressed assets, private path rejection, allowed public/local CORS origins and disallowed-origin rejection. See `http-verification.json`.
- Page opening and Connect do not initialize a worker or download the neural/body model. Sharing requires opt-in and Start.
- One actual browser contribution ran with WebGPU and native FlyBody: 1 simulated second, 34.44 seconds of training wall time including startup, 66 native frames, reward 2.42954. The posture objective **did not pass**. See [browser-verification.md](browser-verification.md).
- Desktop and mobile previews were visually inspected; no browser errors or horizontal overflow. The preview uses 320 × 180 pixels.
- Preserved generation 1 and the eight preexisting results; the real deployment test increased the accepted total to **9**. The next unscored lease was released. At verification: 1 completed job, 7 pending jobs, 0 active leases in generation 1.
- Restarted both services and verified exact checkpoint, model/config hashes and result counts survived. Both services are enabled on boot. See `status-after-restart.json`.
- Original consistent local backup: `coordinator-before-gcp.sqlite3`. Post-launch cloud database backup: `coordinator-after-launch.sqlite3`.

This verifies hosting and contribution plumbing, not successful learning of the complete food-to-flight sequence. The shared candidate remains unverified at the posture stage. Anonymous contributor scores are not execution-attested.

## Contributing

The updated contributor page always shares. Open the live page and press **Start shared training**; it connects automatically. Choose a work/rest budget and preview resolution. Opening the page checks metadata without starting compute. The matching downloaded local client joins the same GCP pool automatically. **Download shared checkpoint** fetches the latest saved candidate; [direct JSON download](https://35.209.223.157/api/training/checkpoint). See [the update verification](../gcp-always-shared/README.md).

## Operations

Use IAP SSH:

```sh
gcloud compute ssh fly-training-1 --project=monereko-20260809 --zone=us-central1-a --tunnel-through-iap
sudo systemctl status fly-training caddy --no-pager
sudo journalctl -u fly-training -u caddy -n 50 --no-pager
```

The packaged installer and update/backup instructions are in [deploy/gcp/README.md](../../deploy/gcp/README.md). Installation preserves an existing database and refuses a model/config mismatch. Database backups use SQLite's online backup API so committed WAL records are included. Keep an off-VM copy; there is no scheduled remote backup service in this deployment.

Expected ongoing hosting is approximately **$3.65/month if the account's eligible VM/disk free allowance is available**, or approximately **$10.16/month without those allowances**, before additional traffic and taxes. These are average-month estimates, not a billing cap. The VM/disk allowance was not audited across the entire billing account. Explicit Standard networking currently includes 200 GiB/month/account outbound traffic, then $0.085/GiB in the first paid band. No GPU charges apply to this server. Stopping the VM still incurs retained disk and reserved IP charges.

Pricing references: [Compute](https://cloud.google.com/products/compute/pricing/general-purpose), [disk](https://cloud.google.com/compute/disks-image-pricing), [IPv4](https://cloud.google.com/vpc/network-pricing), [Standard traffic](https://cloud.google.com/network-tiers/pricing), [Free Tier](https://docs.cloud.google.com/free/docs/free-cloud-features).
