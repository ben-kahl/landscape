# W&B SDK CHANGELOG — relevant fix excerpts

- Source: https://github.com/wandb/wandb/blob/main/CHANGELOG.md
- These are real, verbatim CHANGELOG bullets for login/auth/serialization/artifact/init fixes, with the release they shipped in.

## Fixes by version

- **v0.27.1**: The automations API now supports artifact tag, collection tag, and artifact unlinking events (@tonyyli-wandb in https://github.com/wandb/wandb/pull/11922)
- **v0.26.1**: `Api` methods returning artifacts, registries, automations, and related paginators now accept an optional `start` argument to resume iteration from a saved cursor (@tonyyli-wandb in https://github.com/wandb/wandb/pull/11651)
- **v0.26.1**: Made `wandb.init(id=run_id, reinit="create_new")` raise an error when another run in the same script with the same `run_id` is still running (@timoffex in https://github.com/wandb/wandb/pull/11759)
- **v0.26.0**: JSON serialization and deserialization now use `orjson` for improved performance (@jacobromero in https://github.com/wandb/wandb/pull/11163)
- **v0.26.0**: Fixed `update_automation()` silently dropping event filters (e.g. alias conditions on `OnAddArtifactAlias`) when a new event is provided (@matthoare117-wandb in https://github.com/wandb/wandb/pull/11613)
- **v0.26.0**: Fixed artifact client ID collisions in forked child processes by reseeding the fast ID generator after `fork()` (@tonyyli-wandb in https://github.com/wandb/wandb/pull/11491)
- **v0.26.0**: Fixed deadlock in `artifact.download()` for artifacts with many large files. (@amusipatla-wandb in https://github.com/wandb/wandb/pull/11615)
- **v0.26.0**: Fixed `User.generate_api_key()` failing for users with hashed API keys by using the existing authenticated client instead of querying non-secret key names (@ArtsiomWB in https://github.com/wandb/wandb/pull/11663)
- **v0.25.1**: `ArtifactType.collections()` now supports filtering and ordering of collections. (@amusipatla-wandb in https://github.com/wandb/wandb/pull/11268)
- **v0.25.1**: Warning message when `run.log_artifact` does not create a new version because the artifact content is identical to an existing version. (@pingleiwandb in https://github.com/wandb/wandb/pull/11340)
- **v0.25.1**: `Project.collections()` to fetch filtered and ordered artifact collections in a project. (@amusipatla-wandb in https://github.com/wandb/wandb/pull/11319)
- **v0.25.0**: Sweep agents now exit gracefully when the sweep is deleted, instead of running indefinitely with repeated 404 errors (@domphan-wandb in https://github.com/wandb/wandb/pull/11226)
- **v0.24.2**: wandb.Api() now supports Federated Auth (JWT based authentication). (@ryanbuccellato in https://github.com/wandb/wandb/pull/11243)
- **v0.24.2**: Refresh presigned download url when it expires during artifact file downloads. (@pingleiwandb in https://github.com/wandb/wandb/pull/11242)
- **v0.24.1**: After `wandb login --host <invalid-url>`, using `wandb login --host <valid-url>` works as usual (@timoffex in https://github.com/wandb/wandb/pull/11207)
- **v0.24.0**: Removed the deprecated `wandb.beta.workflows` module, including its `log_model()`, `use_model()`, and `link_model()` functions, and whose modern successors are the `Run.log_artifact`, `Run.use_artifact`, and `Run.link_artifact` methods, respectively (@tonyyli-wandb in [TODO: PR link])
- **v0.24.0**: Fixed `Invalid Client ID digest` error when creating artifacts after calling `random.seed()`. Client IDs could collide when random state was seeded deterministically. (@pingleiwandb in https://github.com/wandb/wandb/pull/11039)
- **v0.24.0**: Fixed CLI error when listing empty artifacts (@ruhiparvatam in https://github.com/wandb/wandb/pull/11157)
- **v0.23.1**: `wandb.Api()` now raises a `UsageError` if `WANDB_IDENTITY_TOKEN_FILE` is set and an explicit API key is not provided (@timoffex in https://github.com/wandb/wandb/pull/10970)
- **v0.23.1**: `wandb.Api()` has only ever worked using an API key
- **v0.23.1**: Anonymous mode, including the `anonymous` setting, the `WANDB_ANONYMOUS` environment variable, `wandb.init(anonymous=...)`, `wandb login --anonymously` and `wandb.login(anonymous=...)` is deprecated and will emit warnings (@timoffex in https://github.com/wandb/wandb/pull/10909)
- **v0.23.1**: `Registry.description` and `ArtifactCollection.description` no longer reject empty strings (@tonyyli-wandb in https://github.com/wandb/wandb/pull/10891)
- **v0.23.1**: Instantiating `Artifact` objects is now significantly faster (@tonyyli-wandb in https://github.com/wandb/wandb/pull/10819)
- **v0.23.1**: Artifact collection aliases are now fetched lazily on accessing `ArtifactCollection.aliases` instead of on instantiating `ArtifactCollection`, improving performance of `Api.artifact_collections()`, `Api.registries().collections()`, etc. (@tonyyli-wandb in https://github.com/wandb/wandb/pull/10731)
- **v0.23.1**: Use explicitly provided API key in `wandb.init(settings=wandb.Settings(api_key="..."))` for login. Use the key from run when logging artifacts via `run.log_artifact` (@pingleiwandb in https://github.com/wandb/wandb/pull/10914)
- **v0.23.1**: Fixed a rare infinite loop in `console_capture.py` (@timoffex in https://github.com/wandb/wandb/pull/10955)
- **v0.23.1**: File upload/download now respects `WANDB_X_EXTRA_HTTP_HEADERS` except for [reference artifacts](https://docs.wandb.ai/models/artifacts/track-external-files) (@pingleiwandb in https://github.com/wandb/wandb/pull/10761)
- **v0.23.0**: `Artifact.files()` now has a correct `len()` when filtering by the `names` parameter (@matthoare117-wandb in https://github.com/wandb/wandb/pull/10796)
- **v0.23.0**: `Artifact.link()` now logs unsaved artifacts instead of raising an error, consistent with the behavior of `Run.link_artifact()` (@tonyyli-wandb in https://github.com/wandb/wandb/10822)
- **v0.23.0**: Logging an artifact with infinite floats in `Artifact.metadata` now raises a `ValueError` early, instead of waiting on request retries to time out (@tonyyli-wandb in https://github.com/wandb/wandb/pull/10845).
