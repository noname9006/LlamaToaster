import { EventEmitter } from "node:events";

// Wakes a worker's long-polling POST /api/worker/queue call the instant a
// job is enqueued or cancelled for it, instead of a fixed-interval poll loop
// re-querying SQLite for every connected worker (MULTIUSER_PLAN.md §1.4) --
// event name is the worker's id, emitted with no payload (listeners re-query
// the DB themselves; the event is only a wake-up signal).
export const queueEvents = new EventEmitter();
queueEvents.setMaxListeners(0); // one long-poll waiter per worker is expected, but never cap it
