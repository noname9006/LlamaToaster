import type { FastifyError, FastifyReply, FastifyRequest } from "fastify";

// App-wide error handler (registered in index.ts). Maps `error.statusCode`
// onto the response. A 4xx is an expected rejection -- e.g. a signed-out
// browser polling an auth-gated route -- so it logs at info without a stack
// trace. Only a 5xx is a real error worth the full error object.
export function appErrorHandler(error: FastifyError, req: FastifyRequest, reply: FastifyReply): void {
  const statusCode = error.statusCode ?? 500;
  if (statusCode >= 500) req.log.error(error);
  else req.log.info({ statusCode }, error.message);
  // Never leak an internal error's message to the client -- it can contain
  // file paths, SQL, or other implementation detail. Only errors we
  // deliberately threw as a 4xx (see errors.ts) have a message meant to be
  // shown to the caller.
  const message = statusCode >= 500 ? "internal server error" : error.message;
  reply.code(statusCode).send({ error: message });
}
