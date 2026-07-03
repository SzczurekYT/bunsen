# AGENTS.md — bunsen

## Toolchain

- **Stable** for everything except `rustfmt` (requires **nightly**).
- MSRV 1.94.1, edition 2024.
- `cargo +nightly fmt` — uses nightly-only rustfmt (`unstable_features = true` in `rustfmt.toml`).
- `cargo clippy --no-deps` — `-D warnings` in CI; clippy `warnings = "deny"` in workspace lints.

## Commands (exact)

```sh
cargo +nightly fmt                          # format (nightly only)
cargo clippy --no-deps                      # lint (warnings = errors)
cargo test --workspace --features=flex       # test (flex backend for CPU)
cargo doc --no-deps --all-features           # docs (CI step)
```

`cargo test --workspace` without `--features=flex` may fail for tests that require a backend. Use the flag.

CI order (`.github/workflows/ci.yml`): fmt (nightly) → clippy → doc → test.

## cargo-make

`Makefile.toml` provides convenience composites (requires `cargo install cargo-make`):

```sh
cargo make ci          # check-format → clippy → test (mirrors CI)
cargo make fix         # clippy-fix → format
cargo make doc         # cargo doc --no-deps
cargo make test        # cargo test --workspace
cargo make book        # mdBook build
cargo make book-serve  # mdBook dev server with live reload
```

Note: `default_to_workspace = false` — tasks do not auto-apply to workspace members.

## Test quirks

- `RUST_TEST_THREADS=8` set in `.cargo/config.toml`.
- `PerformanceBackend` (in `bunsen::support::testing`) is selected by feature flags: `flex` (default), `wgpu`, `cuda`, `metal`.
- Tests and examples that use `burn` tensor ops need a backend feature enabled.
- Some integration tests write temp files (image IO, disk cache).

## Architecture

- **Cargo workspace** with 5 library crates in `crates/` and 9 example binaries in `examples/`.
- All published crates (`bunsen`, `bunsen-contracts-macros`, `bunsen-firehose`, `bunsen-firehose-image`) are **lockstep versioned** (workspace `version = "0.25.0"`).
- `bunsen-preview-chat-dataloader` is part of the workspace but excluded from auto-release (opt-in via `release-plz.toml`).

### crate layout

| Crate | Purpose |
|---|---|
| `bunsen` | Main library: `blocks/` (attention, MLP, conv, patching), `ops/` (arange, noise, clamp, dropout, norm, repeat, conv helpers, embedding), `kits/` (Whisper, Swin), `burner/` (module reflection, optim, init), `contracts/` (tensor shape contracts), `data/` (disk cache) |
| `bunsen-contracts-macros` | `shape_contract![]` proc-macro |
| `bunsen-firehose` | Columnar dataloader / processing pipeline with burn Batcher bridge |
| `bunsen-firehose-image` | Image loading, augmentation, tensor-conversion ops for firehose |
| `bunsen-preview-chat-dataloader` | Arrow-backed chat dataloader with tokenization (unstable/preview) |

### Default features (`bunsen` crate)

`reflection`, `train`, `testing`, `store` — `testing` pulls in `flex` backend.

## Formatting & style conventions

- `fn_params_layout = "Vertical"` (diff-friendly vertical layout).
- `group_imports = "StdExternalCrate"`, `imports_granularity = "Crate"`, `imports_layout = "Vertical"`.
- `reorder_impl_items = true`, `use_field_init_shorthand = true`, `wrap_comments = true`.
- `format_code_in_doc_comments = true`, `format_macro_matchers = true`.
- `#![warn(missing_docs)]` on library crates.
- Tensor shapes in doc comments: `[batch, time, embed]` (single backticks, shape-first phrasing). See `STYLE.md`.

## Commit & release convention

- **Conventional Commits**: `feat:`, `fix:`, `docs:` etc. release-plz derives changelog and version bumps from these.
- `feat!:` or `BREAKING CHANGE:` footer → major bump. `cargo-semver-checks` also promotes API breaks.
- Release is fully automated: merge to `main` → release-plz opens/releases a `chore: release vX.Y.Z` PR.
- Never bump versions or publish by hand.
- Published crates: `bunsen`, `bunsen-contracts-macros`, `bunsen-firehose`, `bunsen-firehose-image` (lockstep).

## Book

- mdBook in `book/` directory. Build: `cargo make book`. Serve: `cargo make book-serve`.
- Preprocessors: mermaid, katex (via `mdbook-katex`).
- Link checker: `mdbook-linkcheck2`.
- The `Makefile.toml` has `ensure-mdbook-suite` meta-task that auto-installs all mdbook tools if missing.

## CI

`.github/workflows/ci.yml` runs on push/PR to `main`:
1. `cargo +nightly fmt -- --check`
2. `cargo clippy --no-deps -- -D warnings`
3. `cargo doc --no-deps --all-features`
4. `cargo test --features=flex`

`.github/workflows/release-plz.yml` automates releases (see CONTRIBUTING.md).

## Gotchas

- Workspace `resolver = "2"` is explicit (avoid additive features for dev-deps).
- `burn` dependency is `0.21.0` — check burn's own backend feature flags when adding backend support.
- The `downloader` crate is used with `default-features = false` (conflicts with `indicatif`).
- Proc-macro crate `bunsen-contracts-macros` has no dev-deps.
- CI skips `bunsen-preview-chat-dataloader` from auto-release (opt-in model).
- GPU backends: enable `wgpu`/`cuda`/`metal` feature on `bunsen` crate for `PerformanceBackend` selection.
