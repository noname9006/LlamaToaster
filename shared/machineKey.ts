// Ed25519 proof of possession for worker re-enrolment (security finding C1).
// Each worker generates a keypair once and keeps the private half in its
// config.json; the server stores the public half on the worker row when the
// machine is approved. A re-enrolment is then approved automatically only if
// the worker signs a fresh server nonce with that key -- knowing machine_id is
// no longer enough.
//
// Both sides build the signed bytes with this one function so they can never
// drift apart. The domain prefix keeps a signature made for this purpose from
// being valid for anything else the same key might ever sign.
export const REENROL_SIGNATURE_DOMAIN = "lt-reenrol-v1";

// TextEncoder rather than Buffer: shared/ is also type-checked by the
// browser builds (client/, admin/), which have no Node types.
export function reenrolMessage(nonce: string, machineId: string): Uint8Array {
  return new TextEncoder().encode(`${REENROL_SIGNATURE_DOMAIN}\n${nonce}\n${machineId}`);
}
