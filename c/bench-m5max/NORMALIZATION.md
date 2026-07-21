# Path normalization (public release)

Machine-local absolute path prefixes in the committed benchmark logs, result
`.json`/`.jsonl` files, and the `validation/epoch-*/` manifests and scripts were
normalized for public release:

| placeholder | meaning |
|---|---|
| `<repo>` / `$REPO` | the repository root (originally a local checkout or build worktree) |
| `<HOME>` / `$HOME` | the user's home directory |

Data files (`.json`, `.jsonl`, `.log`, `.md`) use the literal `<repo>`/`<HOME>`;
shell scripts use the shell-valid `$REPO`/`$HOME`. Only these path prefixes were
rewritten — all measurements, generated output, event ordering, and embedded
content digests (e.g. `target_prompt_sha256`, `generated_text_sha256`) are
unchanged. The frozen `validation/epoch-*/SHA256SUMS` were regenerated to match
the normalized manifests; the recorded engine-binary hashes (`glm`/`ili`
sha256) are unaffected.
