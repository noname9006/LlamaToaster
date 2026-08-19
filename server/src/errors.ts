// Typed HTTP errors a route handler can `throw` directly -- caught by the
// app-wide error handler (server/src/index.ts's setErrorHandler), which maps
// `error.statusCode` onto the actual HTTP response. Route code should prefer
// throwing one of these subclasses over `reply.code(...).send(...)`, so the
// status/message pairing lives in one place instead of being repeated at
// every call site.
export class HttpError extends Error {
  constructor(message: string, readonly statusCode: number) {
    super(message);
    this.name = "HttpError";
  }
}

export class BadRequestError extends HttpError {
  constructor(message = "bad request") {
    super(message, 400);
    this.name = "BadRequestError";
  }
}

export class UnauthorizedError extends HttpError {
  constructor(message = "unauthorized") {
    super(message, 401);
    this.name = "UnauthorizedError";
  }
}

export class ForbiddenError extends HttpError {
  constructor(message = "forbidden") {
    super(message, 403);
    this.name = "ForbiddenError";
  }
}

export class NotFoundError extends HttpError {
  constructor(message = "not found") {
    super(message, 404);
    this.name = "NotFoundError";
  }
}

export class ConflictError extends HttpError {
  constructor(message = "conflict") {
    super(message, 409);
    this.name = "ConflictError";
  }
}
