# API

A representative slice — see `server/src/routes/*.ts` for the full set
(auth, device-flow enrolment, sessions, and admin routes all exist too, but
are meant to be driven by the SPA, not called directly).

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/tests/trigger` | Queue a sweep against one of your own machines (also the optimization flow's `fit`, speed-run and thread-sweep runs) |
| GET | `/api/flows/:flowId` | every run of one New test optimization session |
| GET | `/api/tests/:id/fit-points` | a fit run's llama-fit-params answers plus the machine's K/V cache support rows |
| GET | `/api/tests` | list your own tests |
| GET | `/api/tests/:id` | test detail + results |
| POST | `/api/tests/:id/pause` \| `/resume` \| `/stop` | control a running sweep |
| GET | `/api/models` | list models |
| POST | `/api/models` | register a model |
| GET | `/api/workers` | list your own machines |
| GET | `/api/workers/:id/available-builds` | installed + installable builds for a machine |
| POST | `/api/workers/:id/llama-cpp/install` | download+extract a build (manual trigger) |
| POST | `/api/workers/:id/llama-cpp/activate` | switch a machine's active build |
| DELETE | `/api/workers/:id/llama-cpp/:tag` | remove an installed build (not the active one) |
| GET | `/api/hf/search?q=` | search Hugging Face repos tagged gguf |
| GET | `/api/hf/repo/*` | list `.gguf` files + sizes in a repo (repo id as the wildcard tail) |
| POST | `/api/workers/:id/models/download` | start a download on a machine (returns as soon as it's accepted, not once it finishes) |
| GET | `/api/workers/:id/daemon` | Serve page state: phase, live memory, endpoint settings, API keys (prefixes only) |
| POST | `/api/workers/:id/daemon/load` \| `/unload` | load a model on the machine, or unload it (queued; last request wins) |
| GET | `/api/workers/:id/daemon/models/:modelId/settings` | the remembered load and sampling settings for a model on this machine |
| PUT | `/api/workers/:id/daemon/sampling` | sampling settings, applied on the next request with no reload |
| PUT | `/api/workers/:id/daemon/gateway` | endpoint address, port, allowed origins, autostart (non-loopback needs an API key) |
| POST | `/api/workers/:id/daemon/keys` | create an API key; the response is the only time it is shown |
| DELETE | `/api/workers/:id/daemon/keys/:keyId` | revoke a key; revoking the last one closes the endpoint to the machine itself |
| GET \| DELETE | `/api/workers/:id/daemon/usage?model_id=` | usage profile, speed curve and chart points for a model; delete clears them |
| GET | `/api/results/export` | json \| csv \| md |
| POST | `/api/ai/chat` | AI assistant (SSE streamed) |
| GET | `/api/auth/status` | who's logged in, and whether login is required at all |
