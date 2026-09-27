import { generateKeyPairSync, createPrivateKey, sign } from "node:crypto";
import { reenrolMessage } from "../../shared/machineKey.js";

// This machine's Ed25519 identity key (see shared/machineKey.ts for why it
// exists). PEM strings, stored in config.json next to the session tokens --
// the private half must never be logged.
export interface MachineKeyPair {
  publicKeyPem: string;
  privateKeyPem: string;
}

export function generateMachineKey(): MachineKeyPair {
  const { publicKey, privateKey } = generateKeyPairSync("ed25519");
  return {
    publicKeyPem: publicKey.export({ type: "spki", format: "pem" }).toString(),
    privateKeyPem: privateKey.export({ type: "pkcs8", format: "pem" }).toString(),
  };
}

// base64 Ed25519 signature over the re-enrolment message for this nonce.
export function signReenrol(privateKeyPem: string, nonce: string, machineId: string): string {
  return sign(null, reenrolMessage(nonce, machineId), createPrivateKey(privateKeyPem)).toString("base64");
}
