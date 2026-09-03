# API

A representative slice — see `server/src/routes/*.ts` for the full set
(auth, device-flow enrolment, sessions, and admin routes all exist too, but
are meant to be driven by the SPA, not called directly).

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/runs/trigger` | Queue a sweep against one of your own machines |
| GET | `/api/runs` | list your own runs |
| GET | `/api/runs/:id` | run detail + results |
| POST | `/api/runs/:id/pause` \| `/resume` \| `/stop` | control a running sweep |
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
| GET | `/api/results/export` | json \| csv \| md |
| POST | `/api/ai/chat` | AI assistant (SSE streamed) |
| GET | `/api/auth/status` | who's logged in, and whether login is required at all |
